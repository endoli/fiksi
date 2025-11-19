// Copyright 2025 the Solvi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sparse QR-decomposition implementations.
//!
//! QR-decomposition factors an `m`-by-`n` matrix A into `A = QR`, where the factor `Q` is an
//! orthogonal matrix of size `m`-by-`m` and `R` is an upper-triangular matrix of size `m`-by-`n`,
//! though as `R` is zero under the diagonal, it is often represented as an `n`-by-`n` matrix. This
//! is known as [QR-decomposition][qr].
//!
//! As `Q` is an [orthogonal matrix][orthogonal], the transpose of `Q` is `Q`'s inverse:
//! `Q^T = Q^-1`.
//!
//! This decomposition is useful for linear systems `A x = b` to be solved for `x`.
//! Algebraically,
//!
//! ```text
//!     A x = b
//! => QR x = b
//! =>  R x = Q^T b.
//! ```
//!
//! As `R` is upper-triangular, this can be solved using back-substitution through
//! [`SparseColMat::solve_upper_triangular_mut`]. For over-determined systems, i.e., `m > n`,
//! the first `n` columns of `Q` can be used (as `Q1^T`), and `R x = Q1^T b` minimizes the sum of
//! squared errors `||A x - b||`.
//!
//! [qr]: https://en.wikipedia.org/wiki/QR_decomposition
//! [orthogonal]: https://en.wikipedia.org/wiki/Orthogonal_matrix

use alloc::{vec, vec::Vec};

use crate::{
    PermutationSequence, SparseColMat, SparseColMatStructure,
    decomposition::sparse::cholesky::{self, CholeskyStructure},
    utils,
};

/// The matrix ordering algorithm to use before performing QR-decomposition.
///
/// Forming QR-factors using sparse matrices is faster (and uses less memory) than dense
/// computations only if the resulting factors remain sparse. Entries that were originally zero but
/// change to a non-zero value are called "fill-in." By permuting the matrix, fill-in can be
/// reduced when performing QR-decomposition.
///
/// This type enumerates the supported algorithms for fill-in reduction.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum QrOrdering {
    /// Don't perform any ordering.
    ///
    /// This uses the natural order of the matrix.
    Natural,

    /// Perform [COLAMD](colamd_rs::colamd()) ordering, a fast approximate algorithm for finding a
    /// minimum degree column permutation.
    ///
    /// See "A Column Approximate Minimum Degree Ordering Algorithm" (2004) by Timothy A. Davis et
    /// al.
    Colamd,
}

/// Symbolic QR-decomposition built in preparation for efficient numeric decomposition.
///
/// Build this using [`SymbolicQr::build`], and use the result to perform numerical decomposition
/// using [`SymbolicQr::numeric`] followed by [`Qr::factorize`].
#[derive(Clone, Debug)]
pub struct SymbolicQr {
    row_permutation: Vec<usize>,
    r_structure: SparseColMatStructure,
    h_structure: SparseColMatStructure,

    col_permutation: Vec<usize>,
    inv_col_permutation_sequence: PermutationSequence,
}

impl SymbolicQr {
    /// The structure of the R-factor of this QR-decomposition.
    pub fn r_structure(&self) -> &SparseColMatStructure {
        &self.r_structure
    }
}

/// A reusable struct to perform repeated numeric QR factorizations on matrices with the same
/// sparsity structure.
///
/// First build a [`SymbolicQr`] using the [`SparseColMatStructure`] of the matrices you'll be
/// factorizing, and build this numeric QR struct for a specific type using [`SymbolicQr::build`].
/// Factorize for a concrete [`SparseColMat`] using [`Self::factorize`].
#[derive(Clone, Debug)]
pub struct Qr<'q, T> {
    row_permutation: &'q [usize],
    col_permutation: &'q [usize],
    inv_col_permutation_sequence: &'q PermutationSequence,
    // TODO: as we don't have a concept of `SparseColMatRef`s and such yet, we temporarily take an
    // owned `SparseColMatStructure` here (through an owned `SparseColMat`), so we can solve using
    // the QR-decomposition without cloning.
    r: SparseColMat<T>,
    h_structure: &'q SparseColMatStructure,
    h_values: Vec<T>,
    h_betas: Vec<T>,
    x: Vec<T>,
}

impl SymbolicQr {
    /// Perform symbolic QR-decomposition using the sparsity structure of matrix `a`.
    ///
    /// This finds the structure of the R-factor of the QR-decomposition (see [`CholeskyStructure`]
    /// for the method used here).
    ///
    /// Actual numeric factorization can then be performed, after building the factorization for a
    /// concrete value type using [`SymbolicQr::numeric`], using [`Qr::factorize`].
    ///
    /// # Panics
    ///
    /// This panics if [`QrOrdering::Colamd`] is used and dimensions or the number of nonzeroes of
    /// `a` are larger than [`i32::MAX`].
    pub fn build(a: &SparseColMatStructure, ordering: QrOrdering) -> Self {
        let mut permuted: Option<SparseColMatStructure> = None;

        let (permutation, inv_col_permutation_sequence) = if matches!(ordering, QrOrdering::Colamd)
        {
            let a_len = colamd_rs::colamd_recommended(
                a.row_indices.len().try_into().unwrap(),
                a.nrows().try_into().unwrap(),
                a.ncols().try_into().unwrap(),
            )
            .expect("overflow");
            debug_assert!(
                a_len >= a.row_indices.len(),
                "The returned length should be able to hold at least all indices."
            );

            let mut a_scratch = vec![0; a_len];
            for (a_scratch, a) in a_scratch.iter_mut().zip(&a.row_indices) {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "
                        This does not truncate: for all valid `SparseColMatStructure` `a`, `a`'s
                        row indices are strictly smaller than `a.nrow()`. By the `try_into` above,
                        `a.nrow()` fits into an `i32`.
                    "
                )]
                {
                    *a_scratch = *a as i32;
                }
            }

            #[expect(
                clippy::cast_possible_truncation,
                reason = "
                    This does not truncate: for all valid `SparseColMatStructure` `a`, `a`'s
                    largest column pointer is equal to `a.row_indices.len()`. By the `try_into`
                    above, `a.row_indices.len()` fits into an `i32`.
                "
            )]
            let mut p = Vec::from_iter(a.column_pointers.iter().map(|&p| p as i32));

            colamd_rs::colamd(
                a.nrows().try_into().unwrap(),
                a.ncols().try_into().unwrap(),
                &mut a_scratch,
                &mut p,
                None,
            )
            .expect("valid column ordering");
            let permutation = Vec::from_iter(p.into_iter().map(|p| p as usize).take(a.ncols()));
            permuted = Some(a.permute_columns(&permutation));

            let mut inv_permutation = vec![0; a.ncols()];
            for (idx, &col) in permutation.iter().enumerate() {
                inv_permutation[col] = idx;
            }
            (
                permutation,
                PermutationSequence::build_for_gather_permutation(&inv_permutation),
            )
        } else {
            let permutation = Vec::from_iter(0..a.ncols());
            let permutation_sequence =
                PermutationSequence::build_for_gather_permutation(&permutation);

            (permutation, permutation_sequence)
        };

        // If we've permuted `a` above, use that permutation.
        let a = permuted.as_ref().unwrap_or(a);

        let parents = cholesky::elimination_tree::<false>(a);
        let post = utils::post_order(&parents);
        let counts = cholesky::CholeskyCounts::build(a, &parents, &post);
        let CholeskyStructure {
            l_structure: r_structure,
            row_permutation,
            h_structure,
        } = cholesky::CholeskyStructure::build(a, &parents, &post, &counts);

        Self {
            row_permutation,
            r_structure,
            h_structure,

            col_permutation: permutation,
            inv_col_permutation_sequence,
        }
    }

    /// Prepare for numeric factorization.
    pub fn numeric<'q, T: num_traits::real::Real>(&'q self) -> Qr<'q, T> {
        Qr {
            row_permutation: &self.row_permutation,
            col_permutation: &self.col_permutation,
            inv_col_permutation_sequence: &self.inv_col_permutation_sequence,
            r: SparseColMat {
                structure: self.r_structure.clone(),
                values: vec![T::zero(); self.r_structure.row_indices.len()],
            },
            h_structure: &self.h_structure,
            h_values: vec![T::zero(); self.h_structure.row_indices.len()],
            h_betas: vec![T::zero(); self.h_structure.ncols()],
            x: vec![T::zero(); self.h_structure.nrows()],
        }
    }
}

fn apply_householder<T: num_traits::real::Real>(
    x: &mut [T],
    beta: T,
    householder_rows: &[usize],
    householder_values: &[T],
) {
    let mut tau = T::zero();
    for (idx, &row) in householder_rows.iter().enumerate() {
        tau = tau + householder_values[idx] * x[row];
    }
    tau = tau * beta;
    for (idx, &row) in householder_rows.iter().enumerate() {
        x[row] = x[row] - householder_values[idx] * tau;
    }
}

/// Sparsely calculate a Householder reflector `(I - tau v v^T) y` in place (given sparse `y` in
/// `householder_v`, replacing it with the values for `v`).
fn calculate_householder<T: num_traits::real::Real>(householder_v: &mut [T]) -> (T, T) {
    let beta;
    let norm;

    let sigma = {
        let mut sigma = T::zero();
        for &value in &householder_v[1..] {
            sigma = sigma + value * value;
        }
        sigma
    };

    if T::is_zero(&sigma) {
        norm = householder_v[0].abs();
        if householder_v[0] >= T::zero() {
            beta = T::zero();
        } else {
            beta = T::one() + T::one(); // beta = 2
        }
        householder_v[0] = T::one();
    } else {
        norm = (sigma + householder_v[0] * householder_v[0]).sqrt();
        if householder_v[0] <= T::zero() {
            householder_v[0] = householder_v[0] - norm;
        } else {
            householder_v[0] = -sigma / (householder_v[0] + norm);
        }
        beta = -(norm * householder_v[0]).recip();
    }

    (norm, beta)
}

impl<'q, T: num_traits::real::Real + core::fmt::Debug> Qr<'q, T> {
    /// Perform numeric QR-decomposition of `a`.
    ///
    /// This finds the R-factor.
    pub fn factorize(&mut self, a: &SparseColMat<T>) {
        self.r.values.fill(T::zero());
        self.h_values.fill(T::zero());
        let (_, n) = a.shape();

        for j in 0..n {
            self.x.fill(T::zero());

            let (values, rows) = a.index_column(self.col_permutation[j]);
            for (&val, &row) in values.iter().zip(rows) {
                self.x[self.row_permutation[row]] = val;
            }

            for (idx, &r_row) in self.r.structure.index_column(j).iter().enumerate() {
                if r_row == j {
                    continue;
                }
                let h_range = self.h_structure.column_pointers[r_row]
                    ..self.h_structure.column_pointers[r_row + 1];
                apply_householder(
                    &mut self.x,
                    self.h_betas[r_row],
                    self.h_structure.index_column(r_row),
                    &self.h_values[h_range],
                );
                self.r.values[self.r.structure.column_pointers[j] + idx] = self.x[r_row];
                self.x[r_row] = T::zero();
            }

            for (idx, &row) in self.h_structure.index_column(j).iter().enumerate() {
                self.h_values[self.h_structure.column_pointers[j] + idx] = self.x[row];
                self.x[row] = T::zero();
            }

            let (norm, beta) = calculate_householder(
                &mut self.h_values
                    [self.h_structure.column_pointers[j]..self.h_structure.column_pointers[j + 1]],
            );
            self.h_betas[j] = beta;
            self.r.values[self.r.structure.column_pointers[j + 1] - 1] = norm;
        }
    }

    /// Calculate `Q^T b` in-place where the mutable slice `b` is an `m`-dimensional vector,
    /// overwriting `b`.
    #[inline]
    #[track_caller]
    pub fn q_tr_mul_mut(&self, b: &mut [T]) {
        assert_eq!(
            b.len(),
            self.h_structure.nrows(),
            "The slice must be of length {}, but was {}",
            self.h_structure.nrows(),
            b.len(),
        );
        let mut y = vec![T::zero(); b.len()];
        for i in 0..self.h_structure.nrows() {
            y[self.row_permutation[i]] = b[i];
        }
        for j in 0..self.h_structure.ncols() {
            let range = self.h_structure.index_column_range(j);
            let rows = self.h_structure.index_column(j);
            apply_householder(&mut y, self.h_betas[j], rows, &self.h_values[range]);
        }
        b.copy_from_slice(y.as_slice());
    }

    /// Solve `QR x = b` in place for `x`, overwriting `b`.
    #[inline]
    #[track_caller]
    pub fn solve_mut(&self, b: &mut [T]) -> bool {
        self.q_tr_mul_mut(b);
        let solved = self.r.solve_upper_triangular_mut(&mut b[..self.r.nrows()]);
        self.inv_col_permutation_sequence.permute_slice(b);
        solved
    }

    /// Get a clone of the R-factor of the QR-decomposition.
    ///
    /// Use [`Qr::factorize`] first.
    #[inline]
    pub fn r(&self) -> SparseColMat<T> {
        self.r.clone()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use crate::{SparseColMat, SparseColMatStructure, TripletMat};

    use super::{QrOrdering, SymbolicQr};

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

        let sqr = SymbolicQr::build(&a.structure, QrOrdering::Natural);
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

        let sqr = SymbolicQr::build(&a.structure, QrOrdering::Natural);
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

    #[test]
    fn big_underdetermined_damped() {
        #[expect(
            clippy::approx_constant,
            reason = "Matrix values copied from `fiksi` debug output"
        )]
        let a = SparseColMat::<f64> {
            structure: crate::SparseColMatStructure {
                nrows: 21,
                ncols: 12,
                row_indices: vec![
                    0, 3, 4, 9, 1, 2, 6, 7, 10, 1, 5, 6, 11, 0, 2, 3, 8, 12, 2, 7, 8, 13, 0, 3, 4,
                    14, 0, 1, 4, 5, 15, 1, 5, 6, 16, 0, 2, 3, 8, 17, 0, 1, 4, 5, 18, 1, 2, 6, 7,
                    19, 2, 7, 8, 20,
                ],
                column_pointers: vec![0, 4, 9, 13, 18, 22, 26, 31, 35, 40, 45, 50, 54],
            },
            values: vec![
                -0.639972795218307,
                -0.447215926627107,
                -0.44721355568960997,
                0.7071067811865476,
                -0.4001674913211708,
                -0.3999868083940772,
                -0.8943515981194428,
                0.8944311404385145,
                0.7071067811865476,
                -1.5999173701163563,
                -0.44707388838208617,
                0.44736474932788184,
                0.7071067811865476,
                -0.3999836117142433,
                -0.39992308524211284,
                0.894426025432544,
                -0.8944302168275398,
                0.7071067811865476,
                -1.5998509791693147,
                -0.4472056965355631,
                0.4472075437935279,
                0.7071067811865476,
                0.3199890126711346,
                -0.894426025432544,
                -0.8944272109050886,
                0.7071067811865476,
                -0.1599892158887456,
                0.7999204192698048,
                0.44721355568960997,
                0.44707388838208617,
                0.7071067811865476,
                0.7999715374797847,
                -0.8944970309212447,
                0.8943515981194428,
                0.7071067811865476,
                0.7999620111070526,
                0.7998597000693494,
                0.447215926627107,
                -0.4472075437935279,
                0.7071067811865476,
                0.07999459904310867,
                -0.3998040461586138,
                0.8944272109050886,
                0.8944970309212447,
                0.7071067811865476,
                0.7999969508465515,
                0.7999912790999654,
                -0.44736474932788184,
                0.4472056965355631,
                0.7071067811865476,
                0.79990989363619,
                -0.8944311404385145,
                0.8944302168275398,
                0.7071067811865476,
            ],
        };

        let b = [1.0, 1.5, 3.0, 2.0, -1., -1., 0., 2., 0.5];
        let mut b_aug = [0.; 21];
        b_aug[..9].copy_from_slice(&b);

        let symbolic_qr = SymbolicQr::build(&a.structure, QrOrdering::Natural);

        // The structure of the R-factor of the QR-decomposition of this matrix is as follows.
        //
        //      1   2   3   4   5   6   7   8   9  10  11  12
        // 1  | x           x       x   x       x   x
        // 2  |     x   x   x   x       x   x   x   x   x   x
        // 3  |         x   x   x       x   x   x   x   x   x
        // 4  |             x   x   x   x   x   x   x   x   x
        // 5  |                 x   x   x   x   x   x   x   x
        // 6  |                     x   x   x   x   x   x   x
        // 7  |                         x   x   x   x   x   x
        // 8  |                             x   x   x   x   x
        // 9  |                                 x   x   x   x
        // 10 |                                     x   x   x
        // 11 |                                         x   x
        // 12 |                                             x
        assert_eq!(
            &symbolic_qr.r_structure,
            &SparseColMatStructure {
                nrows: 12,
                ncols: 12,
                row_indices: vec![
                    0, 1, 1, 2, 0, 1, 2, 3, 1, 2, 3, 4, 0, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3,
                    4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3,
                    4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                ],
                column_pointers: vec![0, 1, 2, 4, 8, 12, 16, 23, 30, 39, 49, 59, 70,],
            }
        );

        let mut qr = symbolic_qr.numeric();
        qr.factorize(&a);

        // Its values of structural non-zero entries in column-major order are as follows, but note
        // that signs may differ.
        #[rustfmt::skip]
        let expected_r_values: &[f64] = &[
            1.1443632413010385, 1.5556334124865352, 0.15436383655009123, -1.8536228520871907,
            -0.12585421178785516, 0.10282882662787352, 0.008563258792558315, -1.54707540601071,
            0.15422951447003982, 0.012843745175047827, -0.14469377120662522, 1.8478697696101134,
            0.5201296692321399, 0.5575214776657585, 0.04365561169885357, -1.2724464764782109,
            -0.08529740835127049, -0.20576965296992644, 0.7811288258909915, -0.04377813113172574,
            0.008316988266444225, 0.3008259236524221, -0.9017113054841377, -0.7199558555059773,
            0.19893235872167403, -0.046751894765172206, 0.05504646443179314, -0.018595751423677864,
            0.06702811792357014, -1.4737636307989213, -0.6221416274575114, -0.20566113200307212,
            -0.017126807284168932, -0.06666699005176135, -0.7886693342087231, -0.19739135194411664,
            0.16299197560723072, 0.08071775765390476, 1.0254620935388574, -0.3942754574442847,
            0.10284465535850068, -0.12077520327820984, 0.058923386104347594, -0.0031304307667358033,
            0.47313791434279984, -0.45407284085435595, 0.6647782353923086, -0.045848946149705985,
            -1.0991792174155706, 0.10283743395286726, 0.8070350534372257, 0.21810215706198768,
            -0.7979608124591704, 0.0681845335827414, -0.02924376768962217, -0.14297825088875327,
            0.08760128163160975, 0.17719766185482363, 0.8768683440601007, -0.7199385545174004,
            -0.059954201163681664, 0.6757033310373374, -0.14620625117697145, 0.29104280855826337,
            0.17529529440107028, 0.3210122322891042, 0.022852609822677228, 0.2219222248884028,
            0.10054214844943143, -1.2089860952465823,
        ];

        for (i, expected) in expected_r_values.iter().enumerate() {
            assert!(
                (qr.r.values[i].abs() - expected.abs()).abs() < 1e-8,
                "Expected R's {i}th non-zero value to be {expected}. Got: {}. (Signs are allowed to differ.)",
                qr.r.values[i]
            );
        }

        qr.solve_mut(&mut b_aug);

        let x_expected = [
            -0.5102395940530802,
            0.5431016295284765,
            -0.16962753473158562,
            -0.06340386268226524,
            -0.8165832652859912,
            -0.4547153650418085,
            -0.27695425087745124,
            0.5462601229857467,
            0.801170631112365,
            -0.8552615948837659,
            0.9722340138357437,
            0.2840190700936184,
        ];
        for (i, expected) in x_expected.iter().enumerate() {
            assert!(
                (b_aug[i] - expected).abs() < 1e-8,
                "Expected x[{i}] to be {expected}. Got: {}",
                b_aug[i]
            );
        }

        // Using a column permutation for fill-in reduction, we no longer know the exact structure
        // to expect, but we do expect finding the same solution to the linear system.
        let symbolic_qr = SymbolicQr::build(&a.structure, QrOrdering::Colamd);
        let mut qr = symbolic_qr.numeric();
        qr.factorize(&a);
        let mut b_aug = [0.; 21];
        b_aug[..9].copy_from_slice(&b);
        qr.solve_mut(&mut b_aug);

        for (i, expected) in x_expected.iter().enumerate() {
            assert!(
                (b_aug[i] - expected).abs() < 1e-8,
                "Expected x[{i}] to be {expected}. Got: {}",
                b_aug[i]
            );
        }
    }
}
