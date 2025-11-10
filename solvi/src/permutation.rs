// Copyright 2025 the Solvi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::vec::Vec;
use bitvec::bitvec;

/// A permutation sequence where each index in its range occurs exactly once.
///
/// This can be used to permute a series of elements of length `N` by a sequence of at most `N`
/// in-place swaps.
///
/// # Example
///
/// ```
/// use solvi::PermutationSequence;
///
/// let sentence = &mut ["this", "is", "a", "permutable", "sentence"];
///
/// let p = PermutationSequence::build_for_gather_permutation(&[2, 3, 4, 1, 0]);
/// for swap in p.swaps_iter() {
///     sentence.swap(swap.0, swap.1);
/// }
///
/// assert_eq!(sentence, &["a", "permutable", "sentence", "is", "this"]);
/// ```
#[derive(Clone, Debug)]
pub struct PermutationSequence {
    swap_sequence: Vec<(usize, usize)>,
    indices_len: usize,
}

impl PermutationSequence {
    /// Build a [`PermutationSequence`] for a given `permutation` of indices where each index in
    /// the range `0..permutation.len()` occurs exactly once.
    ///
    /// This builds the sequence such that applying the sequence to some container `a` finds
    /// `p[i] = a[permutation[i]]`.
    ///
    /// It is a logic error for any value in `permutation` to be out of the range
    /// `0..permutation.len()`, and each value must occur exactly once.
    pub fn build_for_gather_permutation(permutation: &[usize]) -> Self {
        let mut swap_sequence = Vec::new();
        let mut seen = bitvec![0; permutation.len()];
        let mut stack = Vec::new();

        for mut i in permutation.iter().copied() {
            while !seen[i] {
                stack.push(i);
                seen.set(i, true);
                i = permutation[i];
            }

            if !stack.is_empty() {
                let pivot = stack[0];
                for swap in stack.drain(..).skip(1).rev() {
                    swap_sequence.push((pivot, swap));
                }
            }
        }

        Self {
            swap_sequence,
            indices_len: permutation.len(),
        }
    }

    /// Permute the elements of a mutable slice in-place.
    #[inline]
    #[track_caller]
    pub fn permute_slice<T>(&self, slice: &mut [T]) {
        assert!(
            slice.len() >= self.indices_len,
            "This permutation sequence is defined for sequences of length {} (was given a sequence of length {})",
            self.indices_len,
            slice.len()
        );
        for swap in self.swaps_iter() {
            slice.swap(swap.0, swap.1);
        }
    }

    /// Iterate over the sequence of swaps, which, when executed on some container of elements,
    /// places those elements in the order defined by this permutation.
    #[inline]
    pub fn swaps_iter(&self) -> impl Iterator<Item = (usize, usize)> {
        self.swap_sequence.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::PermutationSequence;

    #[test]
    fn permutes_slices() {
        for seq in [
            &[] as &[usize],
            &[0],
            &[0, 5, 4, 1, 2, 3],
            &[0, 1, 2, 3, 4],
            &[4, 3, 2, 1, 0],
        ] {
            // permuting a slice of `[0,1,...,seq_len-1]` should result in `seq`.
            let mut index_values = Vec::from_iter(0..seq.len());
            let p = PermutationSequence::build_for_gather_permutation(seq);
            p.permute_slice(&mut index_values);
            assert_eq!(&index_values, seq);
        }
    }

    /// Permuting without anything changing place is a noop.
    #[test]
    fn noop() {
        let p = PermutationSequence::build_for_gather_permutation(&[0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(p.swaps_iter().next().is_none());
    }

    /// Swapping just two pairs of indices touches only those two pairs.
    #[test]
    fn finds_cycles() {
        let p = PermutationSequence::build_for_gather_permutation(&[0, 1, 5, 7, 4, 2, 6, 3]);
        assert_eq!(p.swaps_iter().count(), 2);
    }
}
