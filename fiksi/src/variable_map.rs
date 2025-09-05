// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Mapping of variables to free or fixed values.

use crate::collections::IndexSet;

pub(crate) enum Variable {
    Free {
        /// The value of this variable.
        value: f64,

        /// The index of this variable into the free variables.
        idx: u32,
    },
    Fixed {
        /// The value of this variable.
        value: f64,
    },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct VariableMap<'s> {
    pub(crate) free_variables: &'s IndexSet<u32>,
    pub(crate) variable_values: &'s [f64],
    pub(crate) free_variable_values: &'s [f64],
}

impl VariableMap<'_> {
    /// Get the variable with the global `system_idx` out of the variable map.
    ///
    /// It can be either a free or a fixed variable.
    ///
    /// # Panics
    ///
    /// Panics when `system_idx` is out of bounds.
    pub(crate) fn get_variable(&self, system_idx: u32) -> Variable {
        if let Some(free_idx) = self.free_variables.get_index_of(&system_idx) {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "We don't allow this many variables."
            )]
            Variable::Free {
                value: self.free_variable_values[free_idx],
                idx: free_idx as u32,
            }
        } else {
            Variable::Fixed {
                value: self.variable_values[system_idx as usize],
            }
        }
    }
}
