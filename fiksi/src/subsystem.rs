// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::{collections::btree_map::BTreeMap, vec::Vec};

use crate::{ConstraintId, EncodedConstraint};

pub(crate) struct Subsystem<'s> {
    /// All constraints in the [`crate::System`] this subsystem belongs to.
    all_constraints: &'s [EncodedConstraint],

    /// The constraints that are part of this subsystem. These are indices into
    /// [`Self::all_constraints`].
    constraints: Vec<ConstraintId>,

    /// The indices of free variables.
    free_variables: Vec<u32>,

    /// Map from variable indices to free variable index.
    variable_to_free_variable: BTreeMap<u32, u32>,
}

impl<'s> Subsystem<'s> {
    pub(crate) fn new(
        all_constraints: &'s [EncodedConstraint],
        mut free_variables: Vec<u32>,
        constraints: Vec<ConstraintId>,
    ) -> Self {
        free_variables.sort_unstable();

        let mut variable_to_free_variable = alloc::collections::BTreeMap::new();
        for (idx, &free_variable) in free_variables.iter().enumerate() {
            variable_to_free_variable.insert(
                free_variable,
                idx.try_into().expect("less than 2^32 elements"),
            );
        }

        Self {
            all_constraints,
            free_variables,
            variable_to_free_variable,
            constraints,
        }
    }
}

impl Subsystem<'_> {
    #[inline(always)]
    pub(crate) fn constraints(&self) -> impl ExactSizeIterator<Item = &EncodedConstraint> {
        self.constraints
            .iter()
            .map(|constraint| &self.all_constraints[constraint.id as usize])
    }

    #[inline(always)]
    pub(crate) fn constraint_ids(&self) -> impl ExactSizeIterator<Item = ConstraintId> {
        self.constraints.iter().copied()
    }

    #[inline(always)]
    pub(crate) fn free_variables(&self) -> impl ExactSizeIterator<Item = u32> {
        self.free_variables.iter().copied()
    }

    #[inline(always)]
    pub(crate) fn free_variable_index(&self, variable: u32) -> Option<u32> {
        self.variable_to_free_variable.get(&variable).copied()
    }
}
