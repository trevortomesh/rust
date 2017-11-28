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
use rustc::ty::Ty;
use std::iter::Step;
use std::ops::Range;

pub mod accesses;
pub mod borrows;
pub mod collect;

newtype_index!(PathId { DEBUG_FORMAT = "PathId({})" });

impl Step for PathId {
    fn steps_between(start: &Self, end: &Self) -> Option<usize> {
        Step::steps_between(&start.index(), &end.index())
    }
    fn replace_one(&mut self) -> Self {
        *self = PathId::new(self.index().replace_one());
        *self
    }
    fn replace_zero(&mut self) -> Self {
        *self = PathId::new(self.index().replace_zero());
        *self
    }
    fn add_one(&self) -> Self {
        PathId::new(self.index().add_one())
    }
    fn sub_one(&self) -> Self {
        PathId::new(self.index().sub_one())
    }
    fn add_usize(&self, n: usize) -> Option<Self> {
        self.index().add_usize(n).map(PathId::new)
    }
}

pub struct PathData<'tcx> {
    pub ty: Ty<'tcx>,
    pub last_descendant: PathId,

    /// Whether this path is ever directly accessed,
    /// instead of being just a parent of a path that is.
    // FIXME(eddyb) have a separate notion of "access path",
    // to keep the sets working on it small.
    pub accessed: bool
}

/// A forest of `Place` interior paths into `Local` roots, flattened in
/// pre-order, with each node immediatelly followed by its descendants.
///
/// Paths into dereferences aren't tracked, as they count as distinct
/// "interior" roots, which aren't meaningful without alias analysis.
/// As such, users must handle indirect accesses themselves.
///
/// Paths into array elements aren't currently supported but they could be.
pub struct LocalPaths<'tcx> {
    pub data: IndexVec<PathId, PathData<'tcx>>,
    pub locals: IndexVec<Local, PathId>,
    pub fields: FxHashMap<(PathId, Field), PathId>,
    pub variants: FxHashMap<(PathId, usize), PathId>
}

impl<'tcx> LocalPaths<'tcx> {
    pub fn total_count(&self) -> usize {
        self.data.len()
    }

    pub fn descendants(&self, path: PathId) -> Range<PathId> {
        path.add_one()..self.data[path].last_descendant.add_one()
    }

    pub fn children<'a>(&'a self, path: PathId) -> Children<'a, 'tcx> {
        Children {
            local_paths: self,
            descendants: self.descendants(path)
        }
    }

    /// Obtain the `PathId` for the `elem` component of `base`, if it is tracked.
    pub fn project<V, T>(&self, base: PathId, elem: &ProjectionElem<V, T>) -> Option<PathId> {
        match *elem {
            ProjectionElem::Field(f, _) => self.fields.get(&(base, f)).cloned(),
            ProjectionElem::Downcast(_, v) => self.variants.get(&(base, v)).cloned(),
            // Could support indexing by constants in the future.
            ProjectionElem::ConstantIndex { .. } |
            ProjectionElem::Subslice { .. } => None,
            // Can't support without alias analysis.
            ProjectionElem::Index(_) |
            ProjectionElem::Deref => None
        }
    }

    /// If possible, obtain a `PathId` for the complete `Place` (as `Ok(_)`),
    /// otherwise, give the longest `PathId` prefix (as `Err(Some(_))`).
    pub fn place_path(&self, place: &Place) -> Result<PathId, Option<PathId>> {
        match *place {
            Place::Local(local) => Ok(self.locals[local]),
            Place::Static(_) => Err(None),
            Place::Projection(ref proj) => {
                let base = self.place_path(&proj.base)?;
                match self.project(base, &proj.elem) {
                    Some(child) => Ok(child),
                    None => Err(Some(base))
                }
            }
        }
    }

    /// Like `place_path`, but for the shortest accessed path prefix of the `place`.
    /// If the path doesn't refer to the complete `place`, it's returned in `Err`.
    pub fn place_path_acessed_prefix(&self, place: &Place) -> Result<PathId, Option<PathId>> {
        match *place {
            Place::Local(local) => Ok(self.locals[local]),
            Place::Static(_) => Err(None),
            Place::Projection(ref proj) => {
                let base = self.place_path_acessed_prefix(&proj.base)?;
                if self.data[base].accessed {
                    return Err(Some(base));
                }
                match self.project(base, &proj.elem) {
                    Some(child) => Ok(child),
                    None => Err(Some(base))
                }
            }
        }
    }

    /// Add a new local to the given `mir` (which is assumed to be the
    /// one `self` was created from) and record a new path for it.
    pub fn create_and_record_new_local(&mut self,
                                       mir: &mut Mir<'tcx>,
                                       decl: LocalDecl<'tcx>)
                                       -> Local {
        let path = self.data.push(PathData {
            ty: decl.ty,
            last_descendant: PathId::new(0),
            accessed: true
        });
        self.data[path].last_descendant = path;

        let local = mir.local_decls.push(decl);
        assert_eq!(self.locals.push(path), local);
        local
    }
}

pub struct Children<'a, 'tcx: 'a> {
    local_paths: &'a LocalPaths<'tcx>,
    descendants: Range<PathId>
}

impl<'a, 'tcx> Iterator for Children<'a, 'tcx> {
    type Item = PathId;
    fn next(&mut self) -> Option<PathId> {
        self.descendants.next().map(|child| {
            self.descendants.start = self.local_paths.descendants(child).end;
            child
        })
    }
}
