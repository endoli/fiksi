// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::{collections::btree_map::BTreeMap, vec::Vec};

use crate::Expression;

pub(crate) struct Subsystem<'s> {
    /// All expressions in the [`crate::System`] this subsystem belongs to.
    all_expressions: &'s [Expression],

    /// The expressions that are part of this subsystem. These are indices into
    /// [`Self::all_expressions`].
    expressions: Vec<u32>,

    /// The indices of free variables.
    free_variables: Vec<u32>,

    /// Map from variable indices to free variable index.
    variable_to_free_variable: BTreeMap<u32, u32>,
}

impl<'s> Subsystem<'s> {
    pub(crate) fn new(
        all_expressions: &'s [Expression],
        mut free_variables: Vec<u32>,
        expressions: Vec<u32>,
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
            all_expressions,
            free_variables,
            variable_to_free_variable,
            expressions,
        }
    }
}

impl Subsystem<'_> {
    #[inline(always)]
    pub(crate) fn expressions(&self) -> impl ExactSizeIterator<Item = &Expression> {
        self.expressions
            .iter()
            .copied()
            .map(|expression| &self.all_expressions[expression as usize])
    }

    #[inline(always)]
    pub(crate) fn expression_ids(&self) -> impl ExactSizeIterator<Item = u32> {
        self.expressions.iter().copied()
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
