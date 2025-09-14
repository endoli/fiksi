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

/// Map global system variable indices to free or fixed variables.
///
/// Intended for consumption by, e.g., a numeric optimizer. Having this as trait allows
/// specialization, especially for some common cases that are essentially no-ops (see
/// [`IdentityVariableMap`], which maps indices directly into a slice.
pub(crate) trait VariableMap {
    /// Get the variable with the global `system_idx` out of the variable map.
    ///
    /// The result can be either a free or a fixed variable.
    ///
    /// # Panics
    ///
    /// This may panic when `system_idx` is out of the map's domain.
    fn get_value(&self, system_idx: u32) -> Variable;
}

/// A [`VariableMap`] backed by an [`IndexSet`].
///
/// This maps global system variable indices into either the [`Self::free_variable_values`] slice
/// or [`Self::variable_values`] slice, containing the free and fixed variables respectively.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IndexSetVariableMap<'s> {
    pub(crate) free_variables: &'s IndexSet<u32>,
    pub(crate) variable_values: &'s [f64],
    pub(crate) free_variable_values: &'s [f64],
}

impl VariableMap for IndexSetVariableMap<'_> {
    /// Get the variable with the global `system_idx` out of the variable map.
    ///
    /// It can be either a free or a fixed variable.
    ///
    /// # Panics
    ///
    /// Panics when `system_idx` is out of bounds.
    fn get_value(&self, system_idx: u32) -> Variable {
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

/// A [`VariableMap`],  mapping indices directly to a slice.
///
/// The const-generic parameter `FREE` is true if and only if the variables are considered free.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IdentityVariableMap<'s, const FREE: bool> {
    pub(crate) variable_values: &'s [f64],
}

pub(crate) type IdentityFreeVariableMap<'s> = IdentityVariableMap<'s, true>;

pub(crate) type IdentityFixedVariableMap<'s> = IdentityVariableMap<'s, false>;

impl<const FREE: bool> VariableMap for IdentityVariableMap<'_, FREE> {
    /// Get the variable with the global `system_idx` out of the variable map.
    ///
    /// The variables are free if and only if the const-generic parameter `FREE` is true.
    ///
    /// # Panics
    ///
    /// Panics when `system_idx` is out of bounds.
    fn get_value(&self, system_idx: u32) -> Variable {
        if FREE {
            Variable::Free {
                value: self.variable_values[system_idx as usize],
                idx: system_idx,
            }
        } else {
            Variable::Fixed {
                value: self.variable_values[system_idx as usize],
            }
        }
    }
}
