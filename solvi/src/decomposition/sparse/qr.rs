// Copyright 2025 the Solvi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sparse QR-decomposition implementations.

use alloc::{vec, vec::Vec};

use crate::{SparseColMat, SparseColMatStructure, decomposition::sparse::cholesky, utils};

/// Symbolic QR-decomposition built in preperation for efficient numeric decomposition.
///
/// Build this using [`SymbolicQr::build`], and use the result to perform numerical decomposition
/// using [`SymbolicQr::numeric`] followed by [`Qr::factorize`].
#[derive(Clone, Debug)]
pub struct SymbolicQr {
    r_structure: SparseColMatStructure,
}

impl SymbolicQr {
    /// The structure of the R-factor of this QR-decomposition.
    pub fn r_structure(&self) -> &SparseColMatStructure {
        &self.r_structure
    }
}

/// Reusable QR-decomposition.
#[derive(Clone, Debug)]
pub struct Qr<'q, T> {
    r_structure: &'q SparseColMatStructure,
    r_values: Vec<T>,
    householders: Vec<Householder<T>>,
}

impl SymbolicQr {
    /// Perform symbolic QR-decomposition using the sparsity structure of matrix `a`.
    ///
    /// This finds the structure of the R-factor of the QR-decomposition (see
    /// [`cholesky_l_factor_counts`](crate::decomposition::sparse::cholesky::cholesky_l_factor_counts)
    /// for the method used here).
    ///
    /// Actual numeric factorization can then be performed, after building the factorization for a
    /// concrete value type using [`SymbolicQr::numeric`], using [`Qr::factorize`].
    pub fn build(a: &SparseColMatStructure) -> Self {
        let parents = cholesky::elimination_tree::<false>(a);
        let post = utils::post_order(&parents);
        let counts = cholesky::CholeskyCounts::build(a, &parents, &post);
        let structure = cholesky::CholeskyStructure::build(a, &parents, &post, &counts);

        SymbolicQr {
            r_structure: structure.l_structure,
        }
    }

    /// Prepare for numeric factorization.
    pub fn numeric<'q, T: num_traits::real::Real>(&'q self) -> Qr<'q, T> {
        Qr {
            r_structure: &self.r_structure,
            r_values: vec![T::zero(); self.r_structure.row_indices.len()],
            householders: Vec::with_capacity(self.r_structure.ncols()),
        }
    }
}

impl<'q, T: num_traits::real::Real> Qr<'q, T> {
    /// Perform numeric QR-decomposition of `a`.
    ///
    /// This finds the R-factor.
    pub fn factorize(&mut self, a: &SparseColMat<T>) {
        self.r_values.fill(T::zero());
        self.householders.clear();

        let (m, n) = a.shape();

        // Numeric QR (left-looking Householder)
        let mut y = vec![T::zero(); m];

        for k in 0..n {
            // Scatter column k of `A` into dense `y`.
            y.fill(T::zero());
            let (values, rows) = a.index_column(k);
            for (&val, &row) in values.iter().zip(rows.iter()) {
                y[row] = val;
            }

            // Apply previous Householder reflectors to y (left-looking).
            for reflector in &self.householders {
                reflector.apply(&mut y);
            }

            let start = self.r_structure.column_pointers[k];
            let end = self.r_structure.column_pointers[k + 1];

            for (t, &row) in self.r_structure.row_indices[start..end].iter().enumerate() {
                self.r_values[start + t] = y[row];
            }

            // Build Householder for column k.
            let x0 = y[k];
            let mut sigma = T::zero();
            for &yk in &y[k + 1..] {
                sigma = sigma + yk * yk;
            }

            let (tau, tail, rkk) = if T::is_zero(&sigma) {
                (T::zero(), Vec::new(), x0)
            } else {
                let mut norm = (x0 * x0 + sigma).sqrt();
                if x0 >= T::zero() {
                    norm = -norm;
                }
                let beta = (x0 - norm).recip();
                let mut tail = y[k + 1..].to_vec();
                for t in &mut tail {
                    *t = *t * beta;
                }
                let tau = (norm - x0) / norm;
                (tau, tail, norm)
            };

            // Store diagonal.
            self.r_values[end - 1] = rkk;

            // Store Householder reflector.
            self.householders.push(Householder { k, tau, tail });
        }
    }

    /// Get a clone of the R-factor of the QR-decomposition.
    ///
    /// Use [`Qr::factorize`] first.
    pub fn r(&self) -> SparseColMat<T> {
        SparseColMat {
            structure: self.r_structure.clone(),
            values: self.r_values.clone(),
        }
    }
}

/// Dense, compact Householder for column k.
///
// This represents (I - tau v v^T) y, with v = [0; 1; tail] where the kth
// Householder has k zeros, a leading 1, and then the Householder matrix tail.
#[derive(Debug, Clone)]
pub(crate) struct Householder<T> {
    pub k: usize,
    pub tau: T,
    pub tail: Vec<T>, // length m-k-1
}

impl<T: num_traits::real::Real> Householder<T> {
    /// Column-wise calculation of (I - tau v v^T) y.
    fn apply(&self, y: &mut [T]) {
        if T::is_zero(&self.tau) {
            return;
        }
        let mut dot = y[self.k]; // v^T y starts at head (1 * y[k])
        for (t, &yk) in self.tail.iter().zip(y[self.k + 1..].iter()) {
            dot = dot + *t * yk;
        }
        if T::is_zero(&dot) {
            return;
        }
        y[self.k] = y[self.k] - self.tau * dot;
        for (yk, &t) in y[self.k + 1..].iter_mut().zip(self.tail.iter()) {
            *yk = *yk - self.tau * dot * t;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{SparseColMat, TripletMat};

    use super::SymbolicQr;

    #[test]
    fn underdetermined_damped() {
        // The following sparse matrix
        //      1  2  3
        // 1  | 2     5
        // 2  |       5
        // 3  | 1
        // 4  |    1
        // 5  |       1
        //
        // Should get the following R-factor in the QR-decomposition (though with potential sign
        // differences).
        //
        //      1       2       3
        // 1  | sqrt(5)         2 sqrt(5)
        // 2  |         1
        // 3  |                 sqrt(31)
        //
        let mut triplets = TripletMat::<f64>::new(5, 3);
        triplets.push_triplet(0, 0, 2.);
        triplets.push_triplet(0, 2, 5.);
        triplets.push_triplet(1, 2, 5.);
        triplets.push_triplet(2, 0, 1.);
        triplets.push_triplet(3, 1, 1.);
        triplets.push_triplet(4, 2, 1.);
        let a = SparseColMat::from_triplet_mat(&triplets);

        let sqr = SymbolicQr::build(&a.structure);
        assert_eq!(sqr.r_structure().column_pointers, &[0, 1, 2, 4]);
        assert_eq!(sqr.r_structure().row_indices, &[0, 1, 0, 2]);

        let mut qr = sqr.numeric();
        qr.factorize(&a);
        let r = qr.r();

        let expected_values = [f64::sqrt(5.), 1., 2. * f64::sqrt(5.), f64::sqrt(31.)];

        for (&value, expected_value) in r.values.iter().zip(expected_values) {
            assert!((value.abs() - expected_value).abs() < 1e-8);
        }
    }

    #[test]
    fn underdetermined() {
        // The following sparse matrix
        //      1  2  3
        // 1  | 3  5  3
        // 2  | 1  3  2
        // 3  |
        //
        // Should get the following R-factor in the QR-decomposition (though with potential sign
        // differences).
        //
        //                  1       2       3
        //             1  | 1       9/5     11/10
        // sqrt(10) *  2  |         2/5     3/10
        //             3  |                 0
        //
        // Note that the last diagonal entry is 0. This is expected, as the QR-decomposition makes
        // certain assumptions about both the structural and numeric rank of the input matrix.

        let mut triplets = TripletMat::<f64>::new(5, 3);
        triplets.push_triplet(0, 0, 3.);
        triplets.push_triplet(0, 1, 5.);
        triplets.push_triplet(0, 2, 3.);
        triplets.push_triplet(1, 0, 1.);
        triplets.push_triplet(1, 1, 3.);
        triplets.push_triplet(1, 2, 2.);
        let a = SparseColMat::from_triplet_mat(&triplets);

        let sqr = SymbolicQr::build(&a.structure);
        assert_eq!(sqr.r_structure().column_pointers, &[0, 1, 3, 6]);
        assert_eq!(sqr.r_structure().row_indices, &[0, 0, 1, 0, 1, 2]);

        let mut qr = sqr.numeric();
        qr.factorize(&a);
        let r = qr.r();

        let expected_values = [
            f64::sqrt(10.),
            9. / 5. * f64::sqrt(10.),
            2. / 5. * f64::sqrt(10.),
            11. / 10. * f64::sqrt(10.),
            3. / 10. * f64::sqrt(10.),
            0.,
        ];

        for (&value, expected_value) in r.values.iter().zip(expected_values) {
            assert!((value.abs() - expected_value).abs() < 1e-8);
        }
    }
}
