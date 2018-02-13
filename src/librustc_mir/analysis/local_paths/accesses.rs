// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::indexed_set::IdxSetBuf;
use rustc_data_structures::indexed_vec::Idx;
use rustc::mir::*;
use rustc::mir::visit::{PlaceContext, Visitor};
use rustc::ty;
use syntax::ast;
use analysis::dataflow::{do_dataflow, BitDenotation, BlockSets, DebugFormatted};
use analysis::eventflow::{Backward, Events, EventFlowResults, Forward, PastAndFuture};
use analysis::local_paths::{LocalPaths, PathId};
use analysis::local_paths::borrows::MaybeBorrowed;
use analysis::locations::FlatLocations;

pub struct Accesses<'a> {
    pub results: PastAndFuture<EventFlowResults<'a, Forward, PathId>,
                               EventFlowResults<'a, Backward, PathId>>
}

impl<'a> Accesses<'a> {
    pub fn collect(mir: &Mir,
                   local_paths: &LocalPaths,
                   flat_locations: &'a FlatLocations)
                   -> Self {
        let borrows = ty::tls::with(|tcx| {
            do_dataflow(tcx, mir, ast::DUMMY_NODE_ID, &[],
                        &IdxSetBuf::new_empty(mir.basic_blocks().len()),
                        MaybeBorrowed::new(mir, local_paths),
                        |_, path| DebugFormatted::new(&path))
        });

        let mut collector = AccessPathCollector {
            local_paths,
            location: Location {
                block: START_BLOCK,
                statement_index: !0
            },
            accesses: Events::new(mir, flat_locations, local_paths.total_count()),
            maybe_borrowed: IdxSetBuf::new_empty(0)
        };

        // FIXME(eddyb) introduce a seeker for this (like in eventflow),
        // maybe reusing `dataflow::at_location(::FlowAtLocation)`.
        // That would remove the need for asserting the location.

        for (block, data) in mir.basic_blocks().iter_enumerated() {
            collector.location.block = block;
            collector.maybe_borrowed = borrows.sets().on_entry_set_for(block.index()).to_owned();

            let on_entry = &mut collector.maybe_borrowed.clone();
            let kill_set = &mut collector.maybe_borrowed.clone();
            for (i, statement) in data.statements.iter().enumerate() {
                collector.location.statement_index = i;
                borrows.operator().before_statement_effect(&mut BlockSets {
                    on_entry,
                    kill_set,
                    gen_set: &mut collector.maybe_borrowed,
                }, collector.location);
                // FIXME(eddyb) get rid of temporary with NLL/2phi.
                let location = collector.location;
                collector.visit_statement(block, statement, location);
                borrows.operator().statement_effect(&mut BlockSets {
                    on_entry,
                    kill_set,
                    gen_set: &mut collector.maybe_borrowed,
                }, collector.location);
            }

            if let Some(ref terminator) = data.terminator {
                collector.location.statement_index = data.statements.len();
                borrows.operator().before_terminator_effect(&mut BlockSets {
                    on_entry,
                    kill_set,
                    gen_set: &mut collector.maybe_borrowed,
                }, collector.location);
                // FIXME(eddyb) get rid of temporary with NLL/2phi.
                let location = collector.location;
                collector.visit_terminator(block, terminator, location);
            }
        }
        let results = collector.accesses.flow(mir.args_iter().map(|arg| {
            // All arguments have been accessed prior to the call to this function.
            local_paths.locals[arg]
            // FIXME(eddyb) this should work with just `arg`, not also its descendants.
        }));
        Accesses { results }
    }
}

struct AccessPathCollector<'a, 'b, 'tcx: 'a> {
    local_paths: &'a LocalPaths<'tcx>,
    accesses: Events<'a, 'b, 'tcx, PathId>,
    location: Location,
    maybe_borrowed: IdxSetBuf<PathId>
}

impl<'a, 'b, 'tcx> AccessPathCollector<'a, 'b, 'tcx> {
    fn access_anything_borrowed(&mut self, location: Location) {
        // FIXME(eddyb) OR `maybe_borrowed` into the accesses for performance.
        for path in self.maybe_borrowed.iter() {
            self.accesses.insert_at(path, location);
        }
    }
}

impl<'a, 'b, 'tcx> Visitor<'tcx> for AccessPathCollector<'a, 'b, 'tcx> {
    fn visit_place(&mut self,
                   place: &Place<'tcx>,
                   context: PlaceContext<'tcx>,
                   location: Location) {
        assert_eq!(self.location, location);

        if context.is_use() {
            match self.local_paths.place_path_acessed_prefix(place) {
                Ok(path) | Err(Some(path)) => {
                    self.accesses.insert_at(path, location);
                }
                Err(None) => {}
            }
        }

        // Traverse the projections in `place`.
        let context = if context.is_mutating_use() {
            PlaceContext::Projection(Mutability::Mut)
        } else {
            PlaceContext::Projection(Mutability::Not)
        };
        let mut place = place;
        while let Place::Projection(ref proj) = *place {
            self.visit_projection_elem(&proj.elem, context, location);
            place = &proj.base;
        }
    }

    // Handle the locals used in indexing projections.
    fn visit_local(&mut self,
                   &local: &Local,
                   context: PlaceContext,
                   location: Location) {
        assert_eq!(self.location, location);

        if context.is_use() {
            self.accesses.insert_at(self.local_paths.locals[local], location);
        }
    }

    fn visit_projection_elem(&mut self,
                             elem: &PlaceElem<'tcx>,
                             context: PlaceContext<'tcx>,
                             location: Location) {
        assert_eq!(self.location, location);

        if let ProjectionElem::Deref = *elem {
            self.access_anything_borrowed(location);
        }
        self.super_projection_elem(elem, context, location);
    }

    fn visit_terminator_kind(&mut self,
                             block: BasicBlock,
                             kind: &TerminatorKind<'tcx>,
                             location: Location) {
        assert_eq!(self.location, location);

        match *kind {
            TerminatorKind::Call { .. } => {
                self.access_anything_borrowed(location);
            }
            TerminatorKind::Return => {
                self.visit_local(&RETURN_PLACE, PlaceContext::Move, location);
            }
            _ => {}
        }
        self.super_terminator_kind(block, kind, location);
    }
}
