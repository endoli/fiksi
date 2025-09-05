// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::vec::Vec;

use crate::{Expression, VariableMap, collections::IndexSet};

pub(crate) struct Subsystem<'s> {
    /// The variable values of the full system.
    system_variables: &'s [f64],

    /// All expressions in the [`crate::System`] this subsystem belongs to.
    all_expressions: &'s [Expression],

    /// The expressions that are part of this subsystem. These are indices into
    /// [`Self::all_expressions`].
    expressions: Vec<u32>,

    /// The free variables that are part of this subsystem. These are indices into the system's
    /// variable slice.
    pub(crate) free_variables: IndexSet<u32>,
}

impl<'s> Subsystem<'s> {
    pub(crate) fn new(
        system_variables: &'s [f64],
        all_expressions: &'s [Expression],
        free_variables: impl IntoIterator<Item = u32>,
        expressions: Vec<u32>,
    ) -> Self {
        Self {
            all_expressions,
            expressions,
            free_variables: free_variables.into_iter().collect(),
            system_variables,
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

impl crate::solve::Problem for Subsystem<'_> {
    fn num_variables(&self) -> u32 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "We don't allow this many variables."
        )]
        {
            self.free_variables.len() as u32
        }
    }

    fn num_residuals(&self) -> u32 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "We don't allow this many residuals."
        )]
        {
            self.expressions.len() as u32
        }
    }

    fn calculate_residuals(&mut self, variables: &[f64], residuals: &mut [f64]) {
        let variable_map = VariableMap {
            free_variables: &self.free_variables,
            variable_values: self.system_variables,
            free_variable_values: variables,
        };

        for (row, expression_id) in self.expressions.iter().copied().enumerate() {
            let expression = &self.all_expressions[expression_id as usize];
            residuals[row] = expression.calculate_residual(variable_map);
        }
    }

    fn calculate_residuals_and_jacobian(
        &mut self,
        variables: &[f64],
        residuals: &mut [f64],
        jacobian: &mut [f64],
    ) {
        let variable_map = VariableMap {
            free_variables: &self.free_variables,
            variable_values: self.system_variables,
            free_variable_values: variables,
        };

        for (row, expression_id) in self.expressions.iter().copied().enumerate() {
            let expression = &self.all_expressions[expression_id as usize];
            let gradient = &mut jacobian
                [row * self.free_variables.len()..(row + 1) * self.free_variables.len()];
            residuals[row] = expression.calculate_residual_and_gradient(variable_map, gradient);
        }
    }
}
