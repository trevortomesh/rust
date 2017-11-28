// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use rustc_data_structures::indexed_set::IdxSet;
use rustc::hir;
use rustc::mir::*;
use rustc::mir::visit::{PlaceContext, MutVisitor, Visitor};
use rustc::session::config::FullDebugInfo;
use rustc::ty::TyCtxt;
use std::iter;
use std::mem;
use analysis::eventflow::{After, Before, SparseBitSet};
use analysis::local_paths::{LocalPaths, PathId};
use analysis::local_paths::accesses::Accesses;
use analysis::locations::FlatLocations;
use transform::{MirPass, MirSource};

struct Finder<I: Idx> {
    parent: IndexVec<I, I>,
}

impl<I: Idx> Finder<I> {
    fn find(&mut self, i: I) -> I {
        let parent = self.parent[i];
        if i == parent {
            return i;
        }
        let root = self.find(parent);
        if root != parent {
            self.parent[i] = root;
        }
        root
    }
}

/// Union-find for the points of a binary symmetric relation.
/// Note that the relation is not transitive, only `union` is.
struct UnionFindSymRel<I: Idx> {
    finder: Finder<I>,
    relation: IndexVec<I, SparseBitSet<I>>,
}

impl<I: Idx> UnionFindSymRel<I> {
    fn union(&mut self, a: I, b: I) -> I {
        let a = self.finder.find(a);
        let b = self.finder.find(b);
        if a == b {
            return a;
        }

        let (root, child) = if self.relation[a].capacity() > self.relation[b].capacity() {
            (a, b)
        } else {
            (b, a)
        };
        self.finder.parent[child] = root;

        // Have to juggle the `self.relation` elements as we have
        // no way to borrow two disjoint elements at the same time.
        let child_relation = mem::replace(&mut self.relation[child], SparseBitSet::new());
        // FIXME(eddyb) This could use per-"word" bitwise operations.
        for i in child_relation.iter() {
            // HACK(eddyb) this is really expensive, but used to propagate the relation.
            let i = self.finder.find(i);
            self.relation[root].insert(i);
            self.relation[i].insert(root);
        }
        self.relation[child] = child_relation;

        root
    }

    fn relates(&mut self, a: I, b: I) -> bool {
        let a = self.finder.find(a);
        let b = self.finder.find(b);
        self.relation[a].contains(b) || self.relation[b].contains(a)
    }
}

struct UnionFindWithConflictsAndPriority<I: Idx, P> {
    finder: Finder<I>,
    conflicts: UnionFindSymRel<I>,
    priority: P
}

#[derive(Debug)]
struct Conflict;
impl<I: Idx, P: Fn(I) -> T, T: Ord> UnionFindWithConflictsAndPriority<I, P> {
    fn union(&mut self, a: I, b: I) -> Result<I, Conflict> {
        let a = self.finder.find(a);
        let b = self.finder.find(b);
        if a == b {
            return Ok(a);
        }

        if self.conflicts.relates(a, b) {
            return Err(Conflict);
        }

        let conflict_root = self.conflicts.union(a, b);

        let a_priority = ((self.priority)(a), a == conflict_root);
        let b_priority = ((self.priority)(b), b == conflict_root);
        let (root, child) = if a_priority > b_priority {
            (a, b)
        } else {
            (b, a)
        };
        self.finder.parent[child] = root;

        Ok(root)
    }
}

pub struct UnifyPlaces;

impl MirPass for UnifyPlaces {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          source: MirSource,
                          mir: &mut Mir<'tcx>) {
        // Don't run on constant MIR, because trans might not be able to
        // evaluate the modified MIR.
        // FIXME(eddyb) Remove check after miri is merged.
        let id = tcx.hir.as_local_node_id(source.def_id).unwrap();
        match (tcx.hir.body_owner_kind(id), source.promoted) {
            (_, Some(_)) |
            (hir::BodyOwnerKind::Const, _) |
            (hir::BodyOwnerKind::Static(_), _) => return,

            (hir::BodyOwnerKind::Fn, _) => {
                if tcx.is_const_fn(source.def_id) {
                    // Don't run on const functions, as, again, trans might not be able to evaluate
                    // the optimized IR.
                    return
                }
            }
        }

        // FIXME(eddyb) We should allow multiple user variables
        // per local for debuginfo instead of not optimizing.
        // FIXME(eddyb) (easier) Mark user variables as not renameble,
        // we can always optimize *at least* temporaries!
        if tcx.sess.opts.debuginfo == FullDebugInfo {
            return;
        }
        // HACK(eddyb) testing to debug misoptimization
        // if tcx.sess.opts.debugging_opts.mir_opt_level <= 1 { return; }

        let local_paths = &mut LocalPaths::collect(mir);
        let mut path_to_place: FxHashMap<PathId, Place> = FxHashMap::default();
        let mut dependent_suffix;
        let mut replacement_finder = {
            let flat_locations = &FlatLocations::collect(mir);
            let accesses = &Accesses::collect(mir, local_paths, flat_locations);
            let mut observers = accesses.results.observe();
            let mut conflicts = IndexVec::from_elem_n(SparseBitSet::new(),
                local_paths.total_count());
            let mut candidates = vec![];
            for (block, data) in mir.basic_blocks().iter_enumerated() {
                let mut add_conflicts_at = |past: &IdxSet<_>, future: &IdxSet<_>| {
                    // FIXME(eddyb) use `diff_at_location` (how?) as an optimization.
                    for i in past.iter() {
                        if future.contains(&i) {
                            debug!("unify_places: {:?} live", i);
                            // FIXME(eddyb) Reduce the cost of this already Q_Q.
                            for j in past.iter() {
                                if i != j && future.contains(&j) {
                                    if conflicts[i].insert(j) {
                                        debug!("unify_places:     conflicts with {:?}", j);
                                    }
                                }
                            }
                        }
                    }
                };
                let mut checked_after_last_statement = false;
                for (statement_index, stmt) in data.statements.iter().enumerate() {
                    // FIXME(eddyb) remove this debug! and most of the others.
                    debug!("unify_places: {:?}", stmt);
                    let location = Location { block, statement_index };
                    match stmt.kind {
                        // FIXME(eddyb) figure out how to allow copies.
                        // Maybe if there is any candidate that's a copy,
                        // mark the unification as needing copies? Unclear.
                        // StatementKind::Assign(ref dest, Rvalue::Use(Operand::Copy(ref src))) |
                        StatementKind::Assign(ref dest, Rvalue::Use(Operand::Move(ref src))) => {
                            if let Ok(dest_path) = local_paths.place_path(dest) {
                                if let Ok(src_path) = local_paths.place_path(src) {
                                    candidates.push((dest, dest_path, src, src_path));
                                    if !checked_after_last_statement {
                                        add_conflicts_at(
                                            observers.past.seek(Before(location)),
                                            observers.future.seek(Before(location)));
                                    }
                                    checked_after_last_statement = false;
                                    continue;
                                }
                            }
                        }
                        _ => {}
                    }
                    add_conflicts_at(
                        observers.past.seek(After(location)),
                        observers.future.seek(Before(location)));
                    checked_after_last_statement = true;
                }
                debug!("unify_places: {:?}", data.terminator());
                let location = Location {
                    block,
                    statement_index: data.statements.len()
                };
                add_conflicts_at(
                    observers.past.seek(After(location)),
                    observers.future.seek(Before(location)));
            }

            let mut conflicts = UnionFindSymRel {
                finder: Finder {
                    parent: (0..local_paths.total_count()).map(PathId::new).collect(),
                },
                relation: conflicts
            };

            // FIXME(eddyb) find a better name than `dependent_suffix`.
            dependent_suffix = {
                let mut dependency_collector = DependencyCollector {
                    local_paths,
                    conflicts: &mut conflicts,
                    dependent_suffix: IndexVec::from_elem_n(0, local_paths.total_count())
                };
                dependency_collector.visit_mir(mir);

                // FIXME(eddyb) just pass every local through the dependency collector.

                // Treat every path in the return place and arguments
                // as dependent, to avoid any of them being renamed away.
                for local in iter::once(RETURN_PLACE).chain(mir.args_iter()) {
                    let path = local_paths.locals[local];
                    dependency_collector.dependent_suffix[path] = 1;
                    for descendant in local_paths.descendants(path) {
                        dependency_collector.dependent_suffix[descendant] = 1;
                        dependency_collector.conflicts.union(descendant, path);
                    }
                }

                for local in mir.vars_and_temps_iter() {
                    dependency_collector.collect_path(local_paths.locals[local], &mut vec![]);
                }

                dependency_collector.dependent_suffix
            };

            let mut uf = UnionFindWithConflictsAndPriority {
                finder: Finder {
                    parent: (0..local_paths.total_count()).map(PathId::new).collect(),
                },
                conflicts,
                priority: |path| {
                    // FIXME(eddyb) prefer shorter paths if they're both
                    // the same category (var/arg/return).
                    // That would achieve a form of SROA, maybe?

                    dependent_suffix[path] > 0
                }
            };

            // Union together all the candidate source and targets.
            // Candidates may fail if they could cause a conflict.
            for (a, a_path, b, b_path) in candidates {
                // HACK(eddyb) this is a bit messy, maybe should go into `union`.
                let either_independent =
                    dependent_suffix[uf.finder.find(a_path)] == 0 ||
                    dependent_suffix[uf.finder.find(b_path)] == 0;
                if !either_independent {
                    continue;
                }

                let union_result = uf.union(a_path, b_path);
                debug!("unify_places: candidate ({:?} => {:?}) <- ({:?} => {:?}) unified to {:?}",
                       a_path, a, b_path, b, union_result);

                if let Ok(root) = union_result {
                    // Cache the MIR `Place` representations of paths, for
                    // the paths that we might end up replacing to later.
                    let place = if root == a_path {
                        a
                    } else if root == b_path {
                        b
                    } else {
                        continue;
                    };
                    path_to_place.entry(root).or_insert_with(|| place.clone());
                }
            }

            uf.finder
        };

        // Rename unrenamed independent fields when they would be
        // clobbered from one of their parents being renamed over.
        // FIXME(eddyb) Do we need this? If `x.a` isn't accessed but `x` and `x.a.b` are?
        {
            let mut used_as_rename_destination = SparseBitSet::new();
            for from in PathId::new(0)..PathId::new(local_paths.total_count()) {
                let to = replacement_finder.find(from);
                if from != to {
                    used_as_rename_destination.insert(to);
                }
            }

            for local in mir.vars_and_temps_iter() {
                // FIXME(eddyb) use a range iterator over `used_as_rename_destination`.
                let local_path = local_paths.locals[local];
                let rename_destinations = iter::once(local_path)
                    .chain(local_paths.descendants(local_path))
                    .filter(|&p| used_as_rename_destination.contains(p));
                for destination in rename_destinations {
                    for path in local_paths.descendants(destination) {
                        let independent = dependent_suffix[path] == 0;
                        let renamed = replacement_finder.find(path) != path;
                        if independent && !renamed {
                            // Rename `path` to a a fresh temp to avoid clobbering.
                            let mut decl = mir.local_decls[local].clone();
                            decl.ty = local_paths.data[path].ty;
                            decl.name = None;
                            decl.is_user_variable = false;
                            let new = local_paths.create_and_record_new_local(mir, decl);
                            let new_path = local_paths.locals[new];
                            assert_eq!(new_path, replacement_finder.parent.push(new_path));
                            assert_eq!(new_path, dependent_suffix.push(0));
                            path_to_place.insert(new_path, Place::Local(new));
                            replacement_finder.parent[path] = new_path;

                            debug!("unify_places: de-clobbering {:?} (in {:?}) to \
                                    ({:?} => {:?})",
                                path, local, new_path, new);
                        }
                    }
                }
            }
        }

        // Apply the replacements we computed previously.
        let mut replacer = Replacer {
            local_paths,
            dependent_suffix: &dependent_suffix,
            path_to_place,
            replacement_finder
        };
        replacer.visit_mir(mir);
    }
}

struct DependencyCollector<'a, 'tcx: 'a> {
    local_paths: &'a LocalPaths<'tcx>,
    conflicts: &'a mut UnionFindSymRel<PathId>,
    dependent_suffix: IndexVec<PathId, u32>
}

impl<'a, 'tcx> DependencyCollector<'a, 'tcx> {
    // FIXME(eddyb) Keep track of left/right ranges of variants
    // at every level of parnts and check conflicts between.
    // fields within the current path and lateral parent variants.
    // Note that it appears to be working anyway, because of
    // `Discriminant` and `SetDiscriminant` touching the whole enum.
    fn collect_path(&mut self, path: PathId, prefixes: &mut Vec<PathId>) {
        prefixes.push(path);
        for child in self.local_paths.children(path) {
            self.collect_path(child, prefixes);
        }
        assert_eq!(prefixes.pop(), Some(path));

        // This path is dependent (if at all) on the longest accessed prefix.
        for (i, &parent) in prefixes.iter().rev().enumerate() {
            if self.local_paths.data[parent].accessed {
                debug!("unify_places: {:?} depends on {:?} (suffix length {})",
                       path, parent, i + 1);
                self.dependent_suffix[path] = i as u32 + 1;

                // This path is now dependent on `parent`, so we need
                // all of its conflicts to be reachable from `parent`.
                // FIXME(eddyb) This is symmetric transitivity - maybe
                // we want it to be directional instead? This will
                // happily put `a.x` and `a.y` in the same conflict
                // group (`a`'s`) just because they depend on `a`.
                self.conflicts.union(path, parent);

                break;
            }
        }

    }
}

impl<'a, 'tcx> Visitor<'tcx> for DependencyCollector<'a, 'tcx> {
    fn visit_projection_elem(&mut self,
                             elem: &PlaceElem<'tcx>,
                             context: PlaceContext<'tcx>,
                             location: Location) {
        if let ProjectionElem::Index(i) = *elem {
            let path = self.local_paths.locals[i];
            // Array indices should be integers which have no fields.
            assert_eq!(self.local_paths.descendants(path).count(), 0);
            self.dependent_suffix[path] = 1;
        }
        self.super_projection_elem(elem, context, location);
    }
}

struct Replacer<'a, 'tcx: 'a> {
    local_paths: &'a LocalPaths<'tcx>,
    dependent_suffix: &'a IndexVec<PathId, u32>,
    path_to_place: FxHashMap<PathId, Place<'tcx>>,
    replacement_finder: Finder<PathId>
}

impl<'a, 'tcx> Replacer<'a, 'tcx> {
    fn replace_if_needed(&mut self, place: &mut Place<'tcx>, mut skip: u32) {
        // FIXME(eddyb) use a helper function to get the `PathId`
        // at every level of the `Place` in `O(n)` instead of `O(n^2)`.
        if skip == 0 {
            if let Ok(from) = self.local_paths.place_path(place) {
                let dependent_suffix = self.dependent_suffix[from];
                let to = self.replacement_finder.find(from);

                // This place is fully independent, so we either replace it,
                // or leave alone (even if e.g. its parent might be replaced).
                if dependent_suffix == 0 {
                    if to != from {
                        *place = self.path_to_place[&to].clone();

                        // Recurse in case the replacement is dependent
                        // on a parent that also needs to be replaced.
                        // FIXME(eddyb) precompute this into `path_to_place`.
                        self.replace_if_needed(place, 0);
                    }
                    return;
                }

                assert_eq!(to, from);
                skip = dependent_suffix - 1;
            }
        } else {
            skip -= 1;
        }

        if let Place::Projection(ref mut proj) = *place {
            self.replace_if_needed(&mut proj.base, skip);
        }
    }
}

impl<'a, 'tcx> MutVisitor<'tcx> for Replacer<'a, 'tcx> {
    fn visit_place(&mut self, place: &mut Place<'tcx>, _: PlaceContext<'tcx>, _: Location) {
        self.replace_if_needed(place, 0);
    }

    fn visit_statement(&mut self,
                       block: BasicBlock,
                       statement: &mut Statement<'tcx>,
                       location: Location) {
        // FIXME(eddyb) fuse storage liveness ranges instead of removing them.
        match statement.kind {
            StatementKind::StorageLive(_local) |
            StatementKind::StorageDead(_local) => {
                // FIXME(eddyb) figure out how to even detect relevancy.
                statement.kind = StatementKind::Nop;
            }
            _ => {}
        }

        self.super_statement(block, statement, location);

        // Remove self-assignments resulting from replaced move chains.
        // FIXME(eddyb) do this without comparing `Place`s.
        // (within `super_statement` above we know `PathId`s for both sides)
        let nop = match statement.kind {
            StatementKind::Assign(ref dest, Rvalue::Use(Operand::Copy(ref src))) |
            StatementKind::Assign(ref dest, Rvalue::Use(Operand::Move(ref src))) => {
                dest == src
            }
            _ => false
        };
        if nop {
            statement.kind = StatementKind::Nop;
        }
    }
}
