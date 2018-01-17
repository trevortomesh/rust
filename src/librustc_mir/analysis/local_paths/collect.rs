// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::indexed_vec::{IndexVec, Idx};
use rustc_data_structures::fx::FxHashMap;
use rustc::mir::*;
use rustc::mir::visit::{PlaceContext, Visitor};
use rustc::ty::Ty;
use analysis::local_paths::{LocalPaths, PathData, PathId};

impl<'tcx> LocalPaths<'tcx> {
    pub fn collect(mir: &Mir<'tcx>) -> Self {
        let mut collector = LocalPathCollector {
            locals: mir.local_decls.iter().map(|decl| {
                PathTree::new(decl.ty)
            })
                .collect()
        };
        // All arguments have been accessed prior to the call to this function.
        for arg in mir.args_iter() {
            collector.locals[arg].accessed = true;
        }
        collector.visit_mir(mir);
        let mut fields = FxHashMap::default();
        let mut variants = FxHashMap::default();
        let mut data = IndexVec::new();
        LocalPaths {
            locals: collector.locals.iter().map(|tree| {
                tree.flatten(&mut fields, &mut variants, &mut data)
            }).collect(),
            fields,
            variants,
            data
        }
    }
}

struct PathTree<'tcx> {
    ty: Ty<'tcx>,
    accessed: bool,
    fields: FxHashMap<Field, PathTree<'tcx>>,
    variants: FxHashMap<usize, PathTree<'tcx>>,
}

impl<'tcx> PathTree<'tcx> {
    fn new(ty: Ty<'tcx>) -> Self {
        PathTree {
            ty,
            accessed: false,
            fields: FxHashMap::default(),
            variants: FxHashMap::default()
        }
    }

    fn project(&mut self, elem: &PlaceElem<'tcx>) -> Option<&mut Self> {
        if let Some(adt) = self.ty.ty_adt_def() {
            // Packed types have additional restrictions
            // and it's easier to just not look into them.
            if adt.repr.packed() {
                return None;
            }

            // Enums and unions have overlapping members, so every access
            // of any member must be treated as an access of any other.
            if adt.is_union() || adt.is_enum() {
                self.accessed = true;
            }
        }

        match *elem {
            ProjectionElem::Field(f, ty) => {
                Some(self.fields.entry(f).or_insert(PathTree::new(ty)))
            }
            ProjectionElem::Downcast(_, v) => {
                Some(self.variants.entry(v).or_insert(PathTree::new(self.ty)))
            }
            // Could support indexing by constants in the future.
            ProjectionElem::ConstantIndex { .. } |
            ProjectionElem::Subslice { .. } => None,
            // Can't support without alias analysis.
            ProjectionElem::Index(_) |
            ProjectionElem::Deref => None
        }
    }

    fn flatten(&self,
               fields: &mut FxHashMap<(PathId, Field), PathId>,
               variants: &mut FxHashMap<(PathId, usize), PathId>,
               data: &mut IndexVec<PathId, PathData<'tcx>>)
               -> PathId {
        let root = data.push(PathData {
            ty: self.ty,
            last_descendant: PathId::new(0),
            accessed: self.accessed
        });
        for (&f, child) in &self.fields {
            let child = child.flatten(fields, variants, data);
            fields.insert((root, f), child);
        }
        for (&v, child) in &self.variants {
            let child = child.flatten(fields, variants, data);
            variants.insert((root, v), child);
        }
        data[root].last_descendant = data.last().unwrap();
        root
    }
}

struct LocalPathCollector<'tcx> {
    locals: IndexVec<Local, PathTree<'tcx>>
}

impl<'tcx> LocalPathCollector<'tcx> {
    fn place_path(&mut self, place: &Place<'tcx>) -> Option<&mut PathTree<'tcx>> {
        match *place {
            Place::Local(local) => Some(&mut self.locals[local]),
            Place::Static(_) => None,
            Place::Projection(ref proj) => {
                let base = self.place_path(&proj.base)?;
                base.project(&proj.elem)
            }
        }
    }
}

impl<'tcx> Visitor<'tcx> for LocalPathCollector<'tcx> {
    fn visit_place(&mut self,
                   place: &Place<'tcx>,
                   context: PlaceContext,
                   location: Location) {
        if let Some(path) = self.place_path(place) {
            if context.is_use() {
                path.accessed = true;
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
                   _: Location) {
        if context.is_use() {
            self.locals[local].accessed = true;
        }
    }

    fn visit_terminator_kind(&mut self,
                             block: BasicBlock,
                             kind: &TerminatorKind<'tcx>,
                             location: Location) {
        if let TerminatorKind::Return = *kind {
            self.visit_local(&RETURN_PLACE, PlaceContext::Move, location);
        }
        self.super_terminator_kind(block, kind, location);
    }
}
