// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

/// A fast, small, non-secure pseudorandom number generator.
///
/// This is a 32-bit linear congruential generator with parameters as in Equation 7.1.6 of
/// Numerical Recipes in Fortran second edition, given by D. E. Knuth and H. W. Lewis.
#[derive(Debug)]
pub(crate) struct Rng {
    state: u32,
}

impl Rng {
    /// Create a new `Rng` starting from the given seed.
    ///
    /// No computation is performed.
    #[inline(always)]
    pub(crate) const fn from_seed(seed: u32) -> Self {
        Self { state: seed }
    }

    /// Generate a pseudorandom `u32` distributed uniformly between `0` and `u32::MAX` inclusive.
    #[inline(always)]
    pub(crate) const fn next_u32(&mut self) -> u32 {
        const A: u32 = 1664525;
        const C: u32 = 1013904223;

        self.state = self.state.wrapping_mul(A).wrapping_add(C);
        self.state
    }

    /// Generate a pseudorandom `f64` distributed uniformly between `0` and `1` inclusive.
    ///
    /// Note that are (at best!) 32 bits of randomness here. This is just a utility method.
    #[inline(never)]
    pub(crate) const fn next_f64(&mut self) -> f64 {
        let val = self.next_u32();
        (1. / u32::MAX as f64) * val as f64
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    /// Tests the implementation against a known sequence given in Numerical Recipes in Fortran
    /// second edition, around Equation 7.1.6.
    #[test]
    fn known_sequence() {
        let sequence = [
            0x00000000, 0x3C6EF35F, 0x47502932, 0xD1CCF6E9, 0xAAF95334, 0x6252E503, 0x9F2EC686,
            0x57FE6C2D, 0xA3D95FA8, 0x81FDBEE7, 0x94F0AF1A, 0xCBF633B1,
        ];

        let mut rng = Rng::from_seed(sequence[0]);
        for val in &sequence[1..] {
            assert_eq!(
                rng.next_u32(),
                *val,
                "the random value is not in the known sequence"
            );
        }
    }

    #[test]
    fn floats_in_correct_range() {
        let mut rng = Rng::from_seed(42);

        for _ in 0..32 {
            let val = rng.next_f64();
            assert!(
                (0. ..=1.).contains(&val),
                "the random float must be between 0 and 1 inclusive"
            );
        }
    }
}
