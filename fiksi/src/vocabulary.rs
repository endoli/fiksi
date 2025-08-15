// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vocabulary for some common geometric constructions.

use crate::{ElementHandle, System, constraints, elements};

impl ElementHandle<elements::Line> {
    pub fn create_incident_point(self, system: &mut System) -> ElementHandle<elements::Point> {
        let current = self.get_value(system);
        let p = elements::Point::create(system, current.p0.x, current.p0.y);
        constraints::PointLineIncidence::create(system, p, self);
        p
    }
}

impl ElementHandle<elements::Circle> {
    pub fn create_point_at_center(self, system: &mut System) -> ElementHandle<elements::Point> {
        let current = self.get_value(system);
        let p = elements::Point::create(system, current.center.x, current.center.y);
        constraints::PointCircleCentrality::create(system, p, self);
        p
    }

    pub fn create_incident_point(self, system: &mut System) -> ElementHandle<elements::Point> {
        let current = self.get_value(system);
        let p =
            elements::Point::create(system, current.center.x + current.radius, current.center.y);
        constraints::PointCircleIncidence::create(system, p, self);
        p
    }

    pub fn create_tangent_line(self, system: &mut System) -> ElementHandle<elements::Line> {
        let current = self.get_value(system);
        let p0 = elements::Point::create(
            system,
            current.center.x + current.radius,
            current.center.y - current.radius,
        );
        let p1 = elements::Point::create(
            system,
            current.center.x + current.radius,
            current.center.y + current.radius,
        );
        let line = elements::Line::create(system, p0, p1);
        constraints::LineCircleTangency::create(system, line, self);
        line
    }
}
