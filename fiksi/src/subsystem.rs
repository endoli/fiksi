// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::vec::Vec;

use crate::{Expression, collections::IndexSet};

pub(crate) struct Subsystem<'s> {
    /// All expressions in the [`crate::System`] this subsystem belongs to.
    all_expressions: &'s [Expression],

    /// The expressions that are part of this subsystem. These are indices into
    /// [`Self::all_expressions`].
    expressions: Vec<u32>,

    /// The free variables that are part of this subsystem. These are indices into the system's
    /// variable slice.
    free_variables: IndexSet<u32>,
}

impl<'s> Subsystem<'s> {
    pub(crate) fn new(
        all_expressions: &'s [Expression],
        free_variables: impl IntoIterator<Item = u32>,
        expressions: Vec<u32>,
    ) -> Self {
        Self {
            all_expressions,
            expressions,
            free_variables: free_variables.into_iter().collect(),
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
        #[expect(
            clippy::cast_possible_truncation,
            reason = "We don't allow this many variables."
        )]
        self.free_variables
            .get_index_of(&variable)
            .map(|idx| idx as u32)
    }
}
