// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::num::NonZeroU32;

use alloc::{
    collections::BTreeSet,
    {vec, vec::Vec},
};

use crate::{ConstraintId, elements::element::ElementId};

/// The elements incident to a constraint.
///
/// Currently there are never more than six elements in a constraint, though that may change in the
/// future. For now, it's small enough to store on the stack.
#[derive(Copy, Clone, Debug)]
pub(crate) struct IncidentElements {
    len: u8,
    elements: [ElementId; 6],
}

impl IncidentElements {
    #[inline(always)]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the const panic ensures this never truncates"
    )]
    pub(crate) const fn from_array<const LEN: usize>(elements: [ElementId; LEN]) -> Self {
        const {
            if LEN < 2 || LEN > 6 {
                panic!("`N` must be between 2 and 6 (both inclusive)");
            }
        }

        let mut self_elements = [ElementId { id: 0 }; 6];
        let mut idx = 0;
        while idx < LEN {
            self_elements[idx] = elements[idx];
            idx += 1;
        }
        Self {
            len: LEN as u8,
            elements: self_elements,
        }
    }

    #[inline(always)]
    pub(crate) const fn len(&self) -> usize {
        self.len as usize
    }

    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[ElementId] {
        &self.elements[..self.len()]
    }

    /// Merge multiple elements in this constraint into a single cluster element given by `into`.
    ///
    /// `merge_predicate` should return `false` for at least one of the elements in `self`, to
    /// ensure the new constraint connects at least two elements.
    #[inline(always)]
    pub(crate) fn merge_elements(
        &self,
        merge_predicate: impl Fn(ElementId) -> bool,
        into: ElementId,
    ) -> Self {
        let mut elements = [ElementId { id: 0 }; 6];
        let mut len = 0;
        let mut merged = false;

        for &element in self.as_slice() {
            if merge_predicate(element) {
                if !merged {
                    elements[len] = into;
                    len += 1;
                    merged = true;
                }
            } else {
                elements[len] = element;
                len += 1;
            }
        }

        #[expect(
            clippy::cast_possible_truncation,
            reason = "`len` is always equal to or less than the `self.len`, so this never truncates"
        )]
        Self {
            len: len as u8,
            elements,
        }
    }
}

/// A primitive geometric element.
#[derive(Clone)]
pub(crate) struct Element {
    /// The degrees of freedom of the element.
    pub(crate) dof: i16,

    /// The constraints acting on this element.
    pub(crate) incident_constraints: Vec<ConstraintId>,
}

/// A constraint between elements.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Constraint {
    /// The valency of the constraint (in other words, the degrees of freedom taken away by this
    /// constraint).
    pub(crate) valency: i16,

    /// The elements this constraint acts on.
    pub(crate) incident_elements: IncidentElements,
}

/// A set of elements and the constraints connecting them.
///
/// This represents a connected component, meaning there is a path between all pairs of elements.
#[derive(Clone, Debug, Default)]
pub(crate) struct ConnectedComponent {
    pub(crate) elements: BTreeSet<ElementId>,
    pub(crate) constraints: BTreeSet<ConstraintId>,
}

/// A graph tracking the structure of a geometric constraint system.
///
/// This tracks (primitive) geometric elements and constraints between them, as well as degrees of
/// freedom of elements and the degrees of freedom consumed by constraints.
///
/// Elements are kept organized in separate connected components.
#[derive(Clone)]
pub(crate) struct Graph {
    pub(crate) elements: Vec<Element>,
    pub(crate) constraints: Vec<Constraint>,

    /// The map from elements to connected components.
    ///
    /// The value for an element is `None` iff it is not part of any connected components (i.e., no
    /// constraints connect the element with another element).
    pub(crate) element_connected_component: Vec<Option<NonZeroU32>>,
    pub(crate) connected_components: Vec<ConnectedComponent>,
}

impl Graph {
    pub(crate) fn new() -> Self {
        Self {
            elements: vec![],
            constraints: vec![],
            element_connected_component: vec![],
            connected_components: vec![],
        }
    }

    pub(crate) fn add_element(&mut self, dof: i16) -> ElementId {
        debug_assert!(
            dof >= 0,
            "Elements should have non-negative degrees of freedom"
        );

        let element = ElementId {
            id: self.elements.len().try_into().unwrap(),
        };
        self.elements.push(Element {
            dof,
            incident_constraints: vec![],
        });
        self.element_connected_component.push(None);

        element
    }

    /// Merge all the connected components of the `elements` that are incident to `constraint` into
    /// a single connected component.
    ///
    /// `constraint` and `elements` must be given such that `elements` are the incident elements to
    /// `constraint`.
    fn merge_connected_components(&mut self, constraint: ConstraintId, elements: &[ElementId]) {
        // Find the largest connected component to merge the others in to.
        let target_component_idx = {
            let mut target_component = None;
            let mut size_largest = 0;
            for element in elements {
                if let Some(component_idx) = self.element_connected_component[element.id as usize] {
                    let component = &self.connected_components[component_idx.get() as usize - 1];
                    if component.elements.len() > size_largest {
                        target_component = Some(component_idx);
                        size_largest = component.elements.len();
                    }
                }
            }

            if target_component.is_none() {
                self.connected_components
                    .push(ConnectedComponent::default());
                target_component = Some(
                    u32::try_from(self.connected_components.len())
                        .unwrap()
                        .try_into()
                        .unwrap(),
                );
            }
            target_component.unwrap()
        };

        let mut target_component = core::mem::take(
            &mut self.connected_components[target_component_idx.get() as usize - 1],
        );

        // Merge all elements' components into the target component, and set their component index.
        for element in elements {
            if let Some(component_idx) = self.element_connected_component[element.id as usize] {
                let component = core::mem::take(
                    &mut self.connected_components[component_idx.get() as usize - 1],
                );
                target_component.elements.extend(&component.elements);
                target_component.constraints.extend(&component.constraints);
            } else {
                target_component.elements.insert(*element);
            }
            self.element_connected_component[element.id as usize] = Some(target_component_idx);
        }
        target_component.constraints.insert(constraint);
        self.connected_components[target_component_idx.get() as usize - 1] = target_component;
    }

    fn push_incident_constraint(&mut self, elements: &[ElementId], constraint: ConstraintId) {
        for element in elements {
            self.elements[element.id as usize]
                .incident_constraints
                .push(constraint);
        }
    }

    pub(crate) fn add_constraint(
        &mut self,
        valency: i16,
        incident_elements: IncidentElements,
    ) -> ConstraintId {
        debug_assert!(valency >= 0, "Constraints should have non-negative valency");

        let constraint = ConstraintId {
            id: self.constraints.len().try_into().unwrap(),
        };

        self.merge_connected_components(constraint, incident_elements.as_slice());
        self.push_incident_constraint(incident_elements.as_slice(), constraint);
        self.constraints.push(Constraint {
            valency,
            incident_elements,
        });

        constraint
    }

    pub(crate) fn connected_components(&self) -> &[ConnectedComponent] {
        &self.connected_components
    }
}
