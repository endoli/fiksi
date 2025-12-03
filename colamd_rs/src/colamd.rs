// COLAMD, Copyright (c) 1998-2024, Timothy A. Davis and Stefan Larimore,
// All Rights Reserved.
// Copyright 2025 the Solvi Authors
// SPDX-License-Identifier: BSD-3-Clause

#![expect(
    non_camel_case_types,
    non_snake_case,
    unused_assignments,
    unsafe_op_in_unsafe_fn,
    reason = "transpiled using c2rust"
)]

use alloc::vec;

use crate::options::Options;

#[inline(always)]
fn sqrt(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        f64::sqrt(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrt(x)
    }
}

type int32_t = i32;
type uint32_t = u32;

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Colamd_Row {
    pub(crate) start: int32_t,
    pub(crate) length: int32_t,
    pub(crate) shared1: C2RustUnnamed_0,
    pub(crate) shared2: C2RustUnnamed,
}

#[derive(Copy, Clone, bytemuck::Zeroable)]
#[repr(C)]
union C2RustUnnamed {
    pub(crate) mark: int32_t,
    pub(crate) first_column: int32_t,
}

unsafe impl bytemuck::Pod for C2RustUnnamed {}

#[derive(Copy, Clone, bytemuck::Zeroable)]
#[repr(C)]
union C2RustUnnamed_0 {
    pub(crate) degree: int32_t,
    pub(crate) p: int32_t,
}

unsafe impl bytemuck::Pod for C2RustUnnamed_0 {}

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Colamd_Col {
    pub(crate) start: int32_t,
    pub(crate) length: int32_t,
    pub(crate) shared1: C2RustUnnamed_4,
    pub(crate) shared2: C2RustUnnamed_3,
    pub(crate) shared3: C2RustUnnamed_2,
    pub(crate) shared4: C2RustUnnamed_1,
}

#[derive(Copy, Clone, bytemuck::Zeroable)]
#[repr(C)]
union C2RustUnnamed_1 {
    pub(crate) degree_next: int32_t,
    pub(crate) hash_next: int32_t,
}

unsafe impl bytemuck::Pod for C2RustUnnamed_1 {}

#[derive(Copy, Clone, bytemuck::Zeroable)]
#[repr(C)]
union C2RustUnnamed_2 {
    pub(crate) headhash: int32_t,
    pub(crate) hash: int32_t,
    pub(crate) prev: int32_t,
}

unsafe impl bytemuck::Pod for C2RustUnnamed_2 {}

#[derive(Copy, Clone, bytemuck::Zeroable)]
#[repr(C)]
union C2RustUnnamed_3 {
    pub(crate) score: int32_t,
    pub(crate) order: int32_t,
}

unsafe impl bytemuck::Pod for C2RustUnnamed_3 {}

#[derive(Copy, Clone, bytemuck::Zeroable)]
#[repr(C)]
union C2RustUnnamed_4 {
    pub(crate) thickness: int32_t,
    pub(crate) parent: int32_t,
}

unsafe impl bytemuck::Pod for C2RustUnnamed_4 {}

const __INT_MAX__: core::ffi::c_int = 2147483647 as core::ffi::c_int;
const COLAMD_STATS: core::ffi::c_int = 20 as core::ffi::c_int;
const COLAMD_DENSE_ROW: core::ffi::c_int = 0 as core::ffi::c_int;
const COLAMD_DENSE_COL: core::ffi::c_int = 1 as core::ffi::c_int;
const COLAMD_DEFRAG_COUNT: core::ffi::c_int = 2 as core::ffi::c_int;
const COLAMD_STATUS: core::ffi::c_int = 3 as core::ffi::c_int;
const COLAMD_INFO1: core::ffi::c_int = 4 as core::ffi::c_int;
const COLAMD_INFO2: core::ffi::c_int = 5 as core::ffi::c_int;
const COLAMD_INFO3: core::ffi::c_int = 6 as core::ffi::c_int;
const COLAMD_OK: core::ffi::c_int = 0 as core::ffi::c_int;
const COLAMD_OK_BUT_JUMBLED: core::ffi::c_int = 1 as core::ffi::c_int;
const COLAMD_ERROR_NROW_NEGATIVE: core::ffi::c_int = -(3 as core::ffi::c_int);
const COLAMD_ERROR_NCOL_NEGATIVE: core::ffi::c_int = -(4 as core::ffi::c_int);
const COLAMD_ERROR_NNZ_NEGATIVE: core::ffi::c_int = -(5 as core::ffi::c_int);
const COLAMD_ERROR_P0_NONZERO: core::ffi::c_int = -(6 as core::ffi::c_int);
const COLAMD_ERROR_A_TOO_SMALL: core::ffi::c_int = -(7 as core::ffi::c_int);
const COLAMD_ERROR_COL_LENGTH_NEGATIVE: core::ffi::c_int = -(8 as core::ffi::c_int);
const COLAMD_ERROR_ROW_INDEX_OUT_OF_BOUNDS: core::ffi::c_int = -(9 as core::ffi::c_int);
const INT_MAX: core::ffi::c_int = __INT_MAX__;
const EMPTY: core::ffi::c_int = -(1 as core::ffi::c_int);
const ALIVE: core::ffi::c_int = 0 as core::ffi::c_int;
const DEAD: core::ffi::c_int = -(1 as core::ffi::c_int);
const DEAD_PRINCIPAL: core::ffi::c_int = -(1 as core::ffi::c_int);
const DEAD_NON_PRINCIPAL: core::ffi::c_int = -(2 as core::ffi::c_int);

/// Returns the recommended value of the `a` slice for use by [`crate::colamd`][crate::colamd()].
///
/// Returns `None` if any input argument is negative or arithmetic overflow occurred. The use of
/// this routine is optional. Not needed for [`crate::symamd`], which dynamically allocates its own
/// memory.
#[inline]
pub fn colamd_recommended(nnz: i32, n_row: i32, n_col: i32) -> Option<usize> {
    let nnz = usize::try_from(nnz).ok()?;
    let n_row = usize::try_from(n_row).ok()?;
    let n_col = usize::try_from(n_col).ok()?;

    let c = n_col
        .checked_add(1)?
        .checked_mul(core::mem::size_of::<Colamd_Col>())?
        .wrapping_div(core::mem::size_of::<i32>());
    let r = n_row
        .checked_add(1)?
        .checked_mul(core::mem::size_of::<Colamd_Row>())?
        .wrapping_div(core::mem::size_of::<i32>());

    nnz.checked_mul(2)?
        .checked_add(c)?
        .checked_add(r)?
        .checked_add(n_col)?
        .checked_add(nnz / 5)
}

/// Always inline as the only call-site is [`crate::symamd`].
#[inline(always)]
pub(crate) fn symamd(
    n: i32,
    a: &[i32],
    p: &[i32],
    perm: &mut [i32],
    options: Options,
    stats: &mut [i32; 20],
) -> bool {
    let mut n_row: int32_t = 0;
    let mut nnz: int32_t = 0;
    let mut i: int32_t = 0;
    let mut j: int32_t = 0;
    let mut k: int32_t = 0;
    let mut mnz: int32_t = 0;
    let mut pp: int32_t = 0;
    let mut last_row: int32_t = 0;
    let mut length: int32_t = 0;

    // === Check the input arguments ========================================

    i = 0 as core::ffi::c_int as int32_t;
    while i < COLAMD_STATS as int32_t {
        stats[i as usize] = 0 as core::ffi::c_int as int32_t;
        i += 1;
    }
    stats[COLAMD_STATUS as usize] = COLAMD_OK;
    stats[COLAMD_INFO1 as usize] = -1;
    stats[COLAMD_INFO2 as usize] = -1;
    if n < 0 as int32_t {
        stats[COLAMD_STATUS as usize] = COLAMD_ERROR_NCOL_NEGATIVE;
        stats[COLAMD_INFO1 as usize] = n;
        return false;
    }
    nnz = p[n as usize];
    if nnz < 0 as int32_t {
        stats[COLAMD_STATUS as usize] = COLAMD_ERROR_NNZ_NEGATIVE;
        stats[COLAMD_INFO1 as usize] = nnz;
        return false;
    }
    if p[0] != 0 {
        stats[COLAMD_STATUS as usize] = COLAMD_ERROR_P0_NONZERO;
        stats[COLAMD_INFO1 as usize] = p[0];
        return false;
    }

    // === Allocate count and mark ==========================================

    let mut count = vec![0; n as usize + 1];
    let mut mark = vec![0; n as usize + 1];

    // === Compute column counts of M, check if A is valid ==================

    stats[COLAMD_INFO3 as usize] = 0 as core::ffi::c_int;
    i = 0 as core::ffi::c_int as int32_t;
    while i < n {
        mark[i as usize] = -1;
        i += 1;
    }
    j = 0 as core::ffi::c_int as int32_t;
    while j < n {
        last_row = -(1 as core::ffi::c_int) as int32_t;
        length = p[(j + 1 as int32_t) as usize] - p[j as usize];
        if length < 0 as int32_t {
            // column pointers must be non-decreasing
            stats[COLAMD_STATUS as usize] = COLAMD_ERROR_COL_LENGTH_NEGATIVE as int32_t;
            stats[COLAMD_INFO1 as usize] = j;
            stats[COLAMD_INFO2 as usize] = length;
            return false;
        }
        pp = p[j as usize];
        while pp < p[(j + 1 as int32_t) as usize] {
            i = a[pp as usize];
            if i < 0 as int32_t || i >= n {
                // row index i, in column j, is out of bounds
                stats[COLAMD_STATUS as usize] = COLAMD_ERROR_ROW_INDEX_OUT_OF_BOUNDS;
                stats[COLAMD_INFO1 as usize] = j;
                stats[COLAMD_INFO2 as usize] = i;
                stats[COLAMD_INFO3 as usize] = n;
                return false;
            }

            if i <= last_row || mark[i as usize] == j {
                // row index is unsorted or repeated (or both), thus col is jumbled. This is a
                // notice, not an error condition.
                stats[COLAMD_STATUS as usize] = COLAMD_OK_BUT_JUMBLED;
                stats[COLAMD_INFO1 as usize] = j;
                stats[COLAMD_INFO2 as usize] = i;
                stats[COLAMD_INFO3 as usize] += 1;
            }

            if i > j && mark[i as usize] != j {
                // row k of M will contain column indices i and j
                count[i as usize] += 1;
                count[j as usize] += 1;
            }
            // mark the row as having been seen in this column
            mark[i as usize] = j;
            last_row = i;
            pp += 1;
        }
        j += 1;
    }

    // === Compute column pointers of M =====================================

    // use output permutation, perm, for column pointers of M
    perm[0] = 0;
    j = 1 as core::ffi::c_int as int32_t;
    while j <= n {
        perm[j as usize] = perm[(j - 1 as int32_t) as usize] + count[(j - 1 as int32_t) as usize];
        j += 1;
    }
    j = 0 as core::ffi::c_int as int32_t;
    while j < n {
        count[j as usize] = perm[j as usize];
        j += 1;
    }

    // === Construct M ======================================================

    mnz = perm[n as usize];
    n_row = mnz / 2 as int32_t;
    let m_len = colamd_recommended(mnz, n_row, n).expect("negative inputs or overflow");
    let mut m = vec![0_i32; m_len];
    k = 0 as core::ffi::c_int as int32_t;
    if stats[COLAMD_STATUS as usize] == COLAMD_OK {
        // Matrix is OK
        j = 0 as core::ffi::c_int as int32_t;
        while j < n {
            pp = p[j as usize];
            while pp < p[(j + 1 as int32_t) as usize] {
                i = a[pp as usize];
                if i > j {
                    m[count[i as usize] as usize] = k;
                    count[i as usize] += 1;
                    m[count[j as usize] as usize] = k;
                    count[j as usize] += 1;
                    k += 1;
                }
                pp += 1;
            }
            j += 1;
        }
    } else {
        // Matrix is jumbled. Do not add duplicates to M. Unsorted cols OK.
        i = 0 as core::ffi::c_int as int32_t;
        while i < n {
            mark[i as usize] = -1;
            i += 1;
        }
        j = 0 as core::ffi::c_int as int32_t;
        while j < n {
            pp = p[j as usize];
            while pp < p[(j + 1 as int32_t) as usize] {
                i = a[pp as usize];
                if i > j && mark[i as usize] != j {
                    // row k of M contains column indices i and j
                    m[count[i as usize] as usize] = k;
                    count[i as usize] += 1;
                    m[count[j as usize] as usize] = k;
                    count[j as usize] += 1;
                    k += 1;
                    mark[i as usize] = j;
                }
                pp += 1;
            }
            j += 1;
        }
    }

    // === Adjust the knobs for M ===========================================

    let mut options = options;
    // there are no dense rows in M */
    options.dense_column_control = options.dense_row_control;
    options.dense_row_control = -1.;

    // === Order the columns of M ===========================================
    unsafe {
        colamd(n_row, n, &mut m, perm, options, stats);
    }

    // Note that the output permutation is now in perm

    // === get the statistics for symamd from colamd ========================

    // a dense column in colamd means a dense row and col in symamd
    stats[COLAMD_DENSE_ROW as usize] = stats[COLAMD_DENSE_COL as usize];

    true
}

pub(crate) unsafe fn colamd(
    n_row: i32,
    n_col: i32,
    a: &mut [i32],
    p: &mut [i32],
    options: Options,
    stats: &mut [i32; 20],
) -> bool {
    let mut i: int32_t = 0;
    let mut nnz: int32_t = 0;
    let mut n_col2: int32_t = 0;
    let mut n_row2: int32_t = 0;
    let mut ngarbage: int32_t = 0;
    let mut max_deg: int32_t = 0;

    let stats = stats.as_mut_ptr();
    if stats.is_null() {
        return false;
    }
    i = 0 as core::ffi::c_int as int32_t;
    while i < COLAMD_STATS as int32_t {
        *stats.offset(i as isize) = 0 as core::ffi::c_int as int32_t;
        i += 1;
    }
    *stats.offset(COLAMD_STATUS as isize) = COLAMD_OK as int32_t;
    *stats.offset(COLAMD_INFO1 as isize) = -(1 as core::ffi::c_int) as int32_t;
    *stats.offset(COLAMD_INFO2 as isize) = -(1 as core::ffi::c_int) as int32_t;
    if n_row < 0 as int32_t {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_NROW_NEGATIVE as int32_t;
        *stats.offset(COLAMD_INFO1 as isize) = n_row;
        return false;
    }
    if n_col < 0 as int32_t {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_NCOL_NEGATIVE as int32_t;
        *stats.offset(COLAMD_INFO1 as isize) = n_col;
        return false;
    }
    nnz = p[n_col as usize];
    if nnz < 0 as int32_t {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_NNZ_NEGATIVE as int32_t;
        *stats.offset(COLAMD_INFO1 as isize) = nnz;
        return false;
    }
    if p[0] != 0 as int32_t {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_P0_NONZERO as int32_t;
        *stats.offset(COLAMD_INFO1 as isize) = p[0];
        return false;
    }

    let aggressive = options.aggressive_row_absorption;

    /// Calculate column size and row size in terms of entries in the `i32` buffer, and the total
    /// size needed for the `i32` buffer.
    ///
    /// Returns `None` if the calculation overflowed.
    #[inline]
    fn calculate_sizes(n_col: usize, n_row: usize, nnz: usize) -> Option<(usize, usize, usize)> {
        let col_size = n_col
            .checked_add(1)?
            .checked_mul(core::mem::size_of::<Colamd_Col>())?
            .wrapping_div(core::mem::size_of::<i32>());
        let row_size = n_row
            .checked_add(1)?
            .checked_mul(core::mem::size_of::<Colamd_Row>())?
            .wrapping_div(core::mem::size_of::<i32>());
        let need = nnz
            .checked_mul(2)?
            .checked_add(n_col)?
            .checked_add(col_size)?
            .checked_add(row_size)?;

        Some((col_size, row_size, need))
    }

    let a_len = a.len();
    let Some((col_size, row_size, need)) =
        calculate_sizes(n_col as usize, n_row as usize, nnz as usize)
    else {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_A_TOO_SMALL as int32_t;
        *stats.offset(COLAMD_INFO1 as isize) = 0;
        *stats.offset(COLAMD_INFO2 as isize) = a_len as i32;
        return false;
    };
    if need > a_len {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_A_TOO_SMALL as int32_t;
        *stats.offset(COLAMD_INFO1 as isize) = need as int32_t;
        *stats.offset(COLAMD_INFO2 as isize) = a_len as i32;
        return false;
    }
    let a_len = a_len.wrapping_sub(col_size.wrapping_add(row_size));
    let (a, col) = a.split_at_mut(a_len);
    let (col, row) = col.split_at_mut(col_size);
    let cols: &mut [Colamd_Col] = bytemuck::cast_slice_mut(col);
    let rows: &mut [Colamd_Row] = bytemuck::cast_slice_mut(row);

    if !init_rows_cols(
        n_row,
        n_col,
        rows,
        cols,
        a,
        p,
        &mut *stats.cast::<[i32; 20]>(),
    ) {
        return false;
    }

    let a = a.as_mut_ptr();

    init_scoring(
        n_row,
        n_col,
        rows.as_mut_ptr(),
        cols.as_mut_ptr(),
        a,
        p.as_mut_ptr(),
        &options,
        &mut n_row2,
        &mut n_col2,
        &mut max_deg,
    );
    ngarbage = find_ordering(
        n_row,
        n_col,
        a_len as i32,
        rows.as_mut_ptr(),
        cols.as_mut_ptr(),
        a,
        p.as_mut_ptr(),
        n_col2,
        max_deg,
        2 as int32_t * nnz,
        aggressive as i32,
    );
    order_children(n_col, cols, p);
    *stats.offset(COLAMD_DENSE_ROW as isize) = n_row - n_row2;
    *stats.offset(COLAMD_DENSE_COL as isize) = n_col - n_col2;
    *stats.offset(COLAMD_DEFRAG_COUNT as isize) = ngarbage;

    true
}

/// Initialize rows and columns.
///
/// Takes the column form of the matrix in `A` and creates the row form of the matrix.  Also, row
/// and column attributes are stored in the `cols` and `rows` slices. If the columns are un-sorted
/// or contain duplicate row indices, this routine will also sort and remove duplicate row indices
/// from the column form of the matrix. Returns false if the matrix is invalid, true otherwise.
unsafe fn init_rows_cols(
    n_row: i32,
    n_col: i32,
    rows: &mut [Colamd_Row],
    cols: &mut [Colamd_Col],
    A: &mut [i32],
    p: &mut [i32],
    stats: &mut [i32; 20],
) -> bool {
    assert!(
        rows.len() >= n_row as usize,
        "`rows` must be at least of length `n_row`"
    );
    assert!(
        cols.len() >= n_col as usize,
        "`cols` must be at least of length `n_col`"
    );

    // === Initialize columns, and check column pointers ====================

    for col in 0..n_col {
        (cols[col as usize]).start = p[col as usize];
        (cols[col as usize]).length = p[(col + 1) as usize] - p[col as usize];
        if (cols[col as usize]).length < 0 {
            stats[COLAMD_STATUS as usize] = COLAMD_ERROR_COL_LENGTH_NEGATIVE;
            stats[COLAMD_INFO1 as usize] = col;
            stats[COLAMD_INFO2 as usize] = cols[col as usize].length;
            return false;
        }
        (cols[col as usize]).shared1.thickness = 1;
        (cols[col as usize]).shared2.score = 0;
        (cols[col as usize]).shared3.prev = EMPTY;
        (cols[col as usize]).shared4.degree_next = EMPTY;
    }

    // p [0..n_col] no longer needed, used as "head" in subsequent routines

    // === Scan columns, compute row degrees, and check row indices =========

    stats[COLAMD_INFO3 as usize] = 0; // number of duplicate or unsorted row indices

    for row in rows.iter_mut().take(n_row as usize) {
        row.length = 0;
        row.shared2.mark = -1;
    }
    for col in 0..n_col {
        let mut last_row: i32 = -1;
        for &row in &A[p[col as usize] as usize..p[col as usize + 1] as usize] {
            // make sure row indices within range
            if row < 0 as int32_t || row >= n_row {
                stats[COLAMD_STATUS as usize] = COLAMD_ERROR_ROW_INDEX_OUT_OF_BOUNDS as int32_t;
                stats[COLAMD_INFO1 as usize] = col;
                stats[COLAMD_INFO2 as usize] = row;
                stats[COLAMD_INFO3 as usize] = n_row;
                return false;
            }
            if row <= last_row || rows[row as usize].shared2.mark == col {
                // row index are unsorted or repeated (or both), thus col is jumbled. This is a
                // notice, not an error condition.
                stats[COLAMD_STATUS as usize] = COLAMD_OK_BUT_JUMBLED as int32_t;
                stats[COLAMD_INFO1 as usize] = col;
                stats[COLAMD_INFO2 as usize] = row;
                stats[COLAMD_INFO3 as usize] += 1;
            }
            if (rows[row as usize]).shared2.mark != col {
                rows[row as usize].length += 1;
            } else {
                // this is a repeated entry in the column, it will be removed
                cols[col as usize].length -= 1;
            }
            // mark the row as having been seen in this column
            (rows[row as usize]).shared2.mark = col;
            last_row = row;
        }
    }

    // === Compute row pointers =============================================

    // row form of the matrix starts directly after the column form of matrix in A
    rows[0].start = p[n_col as usize];
    rows[0].shared1.p = (rows[0]).start;
    rows[0].shared2.mark = -1;
    for row in 1..n_row {
        rows[row as usize].start = rows[(row - 1) as usize].start + rows[(row - 1) as usize].length;
        rows[row as usize].shared1.p = (rows[row as usize]).start;
        rows[row as usize].shared2.mark = -1;
    }

    // === Create row form ==================================================

    if stats[COLAMD_STATUS as usize] == COLAMD_OK_BUT_JUMBLED as int32_t {
        // if cols jumbled, watch for repeated row indices
        for col in 0..n_col {
            for cp in p[col as usize]..p[(col + 1) as usize] {
                let row = A[cp as usize];
                if rows[row as usize].shared2.mark != col {
                    A[rows[row as usize].shared1.p as usize] = col;
                    rows[row as usize].shared1.p += 1;
                    rows[row as usize].shared2.mark = col;
                }
            }
        }
    } else {
        // f cols not jumbled, we don't need the mark (this is faster)
        for col in 0..n_col {
            for cp in p[col as usize]..p[(col + 1) as usize] {
                let row = A[cp as usize];
                A[rows[row as usize].shared1.p as usize] = col;
                rows[row as usize].shared1.p += 1;
            }
        }
    }

    // === Clear the row marks and set row degrees ==========================

    for row in rows.iter_mut().take(n_row as usize) {
        row.shared2.mark = 0;
        row.shared1.degree = row.length;
    }

    // === See if we need to re-create columns ==============================

    if stats[COLAMD_STATUS as usize] == COLAMD_OK_BUT_JUMBLED as int32_t {
        // === Compute col pointers =========================================

        // col form of the matrix starts at `a[0]`. Note, we may have a gap between the col form and
        // the row form if there were duplicate entries, if so, it will be removed upon the first
        // garbage collection

        cols[0].start = 0;
        p[0] = cols[0].start;
        for col in 1..n_col {
            // note that the lengths here are for pruned columns, i.e. no duplicate row indices
            // will exist for these columns
            (cols[col as usize]).start =
                cols[(col - 1) as usize].start + cols[(col - 1) as usize].length;
            p[col as usize] = cols[col as usize].start;
        }

        // === Re-create col form ===========================================

        for row in 0..n_row {
            let row_ = rows[row as usize];
            for rp in A[row_.start as usize]..A[row_.start as usize] + row_.length {
                A[p[rp as usize] as usize] = row;
                p[rp as usize] += 1;
            }
        }
    }

    // === Done.  Matrix is not (or no longer) jumbled ======================

    true
}
unsafe extern "C" fn init_scoring(
    n_row: int32_t,
    n_col: int32_t,
    Row: *mut Colamd_Row,
    Col: *mut Colamd_Col,
    A: *mut int32_t,
    head: *mut int32_t,
    options: &Options,
    p_n_row2: *mut int32_t,
    p_n_col2: *mut int32_t,
    p_max_deg: *mut int32_t,
) {
    let mut c: int32_t = 0;
    let mut r: int32_t = 0;
    let mut row: int32_t = 0;
    let mut cp: *mut int32_t = core::ptr::null_mut::<int32_t>();
    let mut deg: int32_t = 0;
    let mut cp_end: *mut int32_t = core::ptr::null_mut::<int32_t>();
    let mut new_cp: *mut int32_t = core::ptr::null_mut::<int32_t>();
    let mut col_length: int32_t = 0;
    let mut score: int32_t = 0;
    let mut n_col2: int32_t = 0;
    let mut n_row2: int32_t = 0;
    let mut dense_row_count: int32_t = 0;
    let mut dense_col_count: int32_t = 0;
    let mut min_score: int32_t = 0;
    let mut max_deg: int32_t = 0;
    let mut next_col: int32_t = 0;
    if options.dense_row_control < 0. {
        dense_row_count = n_col - 1 as int32_t;
    } else {
        dense_row_count =
            (if 16.0f64 > options.dense_row_control * sqrt(n_col as core::ffi::c_double) {
                16.0f64
            } else {
                options.dense_row_control * sqrt(n_col as core::ffi::c_double)
            }) as int32_t;
    }
    if options.dense_column_control < 0. {
        dense_col_count = n_row - 1 as int32_t;
    } else {
        dense_col_count = (if 16.0f64
            > options.dense_column_control
                * sqrt((if n_row < n_col { n_row } else { n_col }) as core::ffi::c_double)
        {
            16.0f64
        } else {
            options.dense_column_control
                * sqrt((if n_row < n_col { n_row } else { n_col }) as core::ffi::c_double)
        }) as int32_t;
    }
    max_deg = 0 as core::ffi::c_int as int32_t;
    n_col2 = n_col;
    n_row2 = n_row;
    c = n_col - 1 as int32_t;
    while c >= 0 as int32_t {
        deg = (*Col.offset(c as isize)).length;
        if deg == 0 as int32_t {
            n_col2 -= 1;
            (*Col.offset(c as isize)).shared2.order = n_col2;
            (*Col.offset(c as isize)).start = DEAD_PRINCIPAL as int32_t;
        }
        c -= 1;
    }
    c = n_col - 1 as int32_t;
    while c >= 0 as int32_t {
        if (*Col.offset(c as isize)).start >= ALIVE as int32_t {
            deg = (*Col.offset(c as isize)).length;
            if deg > dense_col_count {
                n_col2 -= 1;
                (*Col.offset(c as isize)).shared2.order = n_col2;
                cp = &mut *A.offset((*Col.offset(c as isize)).start as isize) as *mut int32_t;
                cp_end = cp.offset((*Col.offset(c as isize)).length as isize);
                while cp < cp_end {
                    (*Row.offset(*cp as isize)).shared1.degree -= 1;
                    cp = cp.offset(1);
                }
                (*Col.offset(c as isize)).start = DEAD_PRINCIPAL as int32_t;
            }
        }
        c -= 1;
    }
    r = 0 as core::ffi::c_int as int32_t;
    while r < n_row {
        deg = (*Row.offset(r as isize)).shared1.degree;
        if deg > dense_row_count || deg == 0 as int32_t {
            (*Row.offset(r as isize)).shared2.mark = DEAD as int32_t;
            n_row2 -= 1;
        } else {
            max_deg = if max_deg > deg { max_deg } else { deg };
        }
        r += 1;
    }
    c = n_col - 1 as int32_t;
    while c >= 0 as int32_t {
        if (*Col.offset(c as isize)).start >= ALIVE as int32_t {
            score = 0 as core::ffi::c_int as int32_t;
            cp = &mut *A.offset((*Col.offset(c as isize)).start as isize) as *mut int32_t;
            new_cp = cp;
            cp_end = cp.offset((*Col.offset(c as isize)).length as isize);
            while cp < cp_end {
                let fresh26 = cp;
                cp = cp.offset(1);
                row = *fresh26;
                if (*Row.offset(row as isize)).shared2.mark < ALIVE as int32_t {
                    continue;
                }
                let fresh27 = new_cp;
                new_cp = new_cp.offset(1);
                *fresh27 = row;
                score = (score as core::ffi::c_int
                    + ((*Row.offset(row as isize)).shared1.degree - 1 as int32_t)
                        as core::ffi::c_int) as int32_t;
                score = if score < n_col { score } else { n_col };
            }
            col_length = new_cp.offset_from(
                &mut *A.offset((*Col.offset(c as isize)).start as isize) as *mut int32_t,
            ) as core::ffi::c_long as int32_t;
            if col_length == 0 as int32_t {
                n_col2 -= 1;
                (*Col.offset(c as isize)).shared2.order = n_col2;
                (*Col.offset(c as isize)).start = DEAD_PRINCIPAL as int32_t;
            } else {
                (*Col.offset(c as isize)).length = col_length;
                (*Col.offset(c as isize)).shared2.score = score;
            }
        }
        c -= 1;
    }
    c = 0 as core::ffi::c_int as int32_t;
    while c <= n_col {
        *head.offset(c as isize) = EMPTY as int32_t;
        c += 1;
    }
    min_score = n_col;
    c = n_col - 1 as int32_t;
    while c >= 0 as int32_t {
        if (*Col.offset(c as isize)).start >= ALIVE as int32_t {
            score = (*Col.offset(c as isize)).shared2.score;
            next_col = *head.offset(score as isize);
            (*Col.offset(c as isize)).shared3.prev = EMPTY as int32_t;
            (*Col.offset(c as isize)).shared4.degree_next = next_col;
            if next_col != EMPTY as int32_t {
                (*Col.offset(next_col as isize)).shared3.prev = c;
            }
            *head.offset(score as isize) = c;
            min_score = if min_score < score { min_score } else { score };
        }
        c -= 1;
    }
    *p_n_col2 = n_col2;
    *p_n_row2 = n_row2;
    *p_max_deg = max_deg;
}
unsafe extern "C" fn find_ordering(
    n_row: int32_t,
    n_col: int32_t,
    Alen: int32_t,
    Row: *mut Colamd_Row,
    Col: *mut Colamd_Col,
    A: *mut int32_t,
    head: *mut int32_t,
    n_col2: int32_t,
    mut max_deg: int32_t,
    mut pfree: int32_t,
    aggressive: int32_t,
) -> int32_t {
    let mut k: int32_t = 0;
    let mut pivot_col: int32_t = 0;
    let mut cp: *mut int32_t = core::ptr::null_mut::<int32_t>();
    let mut rp: *mut int32_t = core::ptr::null_mut::<int32_t>();
    let mut pivot_row: int32_t = 0;
    let mut new_cp: *mut int32_t = core::ptr::null_mut::<int32_t>();
    let mut new_rp: *mut int32_t = core::ptr::null_mut::<int32_t>();
    let mut pivot_row_start: int32_t = 0;
    let mut pivot_row_degree: int32_t = 0;
    let mut pivot_row_length: int32_t = 0;
    let mut pivot_col_score: int32_t = 0;
    let mut needed_memory: int32_t = 0;
    let mut cp_end: *mut int32_t = core::ptr::null_mut::<int32_t>();
    let mut rp_end: *mut int32_t = core::ptr::null_mut::<int32_t>();
    let mut row: int32_t = 0;
    let mut col: int32_t = 0;
    let mut max_score: int32_t = 0;
    let mut cur_score: int32_t = 0;
    let mut hash: uint32_t = 0;
    let mut head_column: int32_t = 0;
    let mut first_col: int32_t = 0;
    let mut tag_mark: int32_t = 0;
    let mut row_mark: int32_t = 0;
    let mut set_difference: int32_t = 0;
    let mut min_score: int32_t = 0;
    let mut col_thickness: int32_t = 0;
    let mut max_mark: int32_t = 0;
    let mut pivot_col_thickness: int32_t = 0;
    let mut prev_col: int32_t = 0;
    let mut next_col: int32_t = 0;
    let mut ngarbage: int32_t = 0;
    max_mark = INT_MAX as int32_t - n_col;
    tag_mark = clear_mark(0 as int32_t, max_mark, n_row, Row);
    min_score = 0 as core::ffi::c_int as int32_t;
    ngarbage = 0 as core::ffi::c_int as int32_t;
    k = 0 as core::ffi::c_int as int32_t;
    while k < n_col2 {
        while *head.offset(min_score as isize) == EMPTY as int32_t && min_score < n_col {
            min_score += 1;
        }
        pivot_col = *head.offset(min_score as isize);
        next_col = (*Col.offset(pivot_col as isize)).shared4.degree_next;
        *head.offset(min_score as isize) = next_col;
        if next_col != EMPTY as int32_t {
            (*Col.offset(next_col as isize)).shared3.prev = EMPTY as int32_t;
        }
        pivot_col_score = (*Col.offset(pivot_col as isize)).shared2.score;
        (*Col.offset(pivot_col as isize)).shared2.order = k;
        pivot_col_thickness = (*Col.offset(pivot_col as isize)).shared1.thickness;
        k += pivot_col_thickness;
        needed_memory = if pivot_col_score < n_col - k {
            pivot_col_score
        } else {
            n_col - k
        };
        if pfree + needed_memory >= Alen {
            pfree = garbage_collection(n_row, n_col, Row, Col, A, &mut *A.offset(pfree as isize));
            ngarbage += 1;
            tag_mark = clear_mark(0 as int32_t, max_mark, n_row, Row);
        }
        pivot_row_start = pfree;
        pivot_row_degree = 0 as core::ffi::c_int as int32_t;
        (*Col.offset(pivot_col as isize)).shared1.thickness = -pivot_col_thickness;
        cp = &mut *A.offset((*Col.offset(pivot_col as isize)).start as isize) as *mut int32_t;
        cp_end = cp.offset((*Col.offset(pivot_col as isize)).length as isize);
        while cp < cp_end {
            let fresh28 = cp;
            cp = cp.offset(1);
            row = *fresh28;
            if (*Row.offset(row as isize)).shared2.mark >= ALIVE as int32_t {
                rp = &mut *A.offset((*Row.offset(row as isize)).start as isize) as *mut int32_t;
                rp_end = rp.offset((*Row.offset(row as isize)).length as isize);
                while rp < rp_end {
                    let fresh29 = rp;
                    rp = rp.offset(1);
                    col = *fresh29;
                    col_thickness = (*Col.offset(col as isize)).shared1.thickness;
                    if col_thickness > 0 as int32_t
                        && (*Col.offset(col as isize)).start >= ALIVE as int32_t
                    {
                        (*Col.offset(col as isize)).shared1.thickness = -col_thickness;
                        let fresh30 = pfree;
                        pfree += 1;
                        *A.offset(fresh30 as isize) = col;
                        pivot_row_degree += col_thickness;
                    }
                }
            }
        }
        (*Col.offset(pivot_col as isize)).shared1.thickness = pivot_col_thickness;
        max_deg = if max_deg > pivot_row_degree {
            max_deg
        } else {
            pivot_row_degree
        };
        cp = &mut *A.offset((*Col.offset(pivot_col as isize)).start as isize) as *mut int32_t;
        cp_end = cp.offset((*Col.offset(pivot_col as isize)).length as isize);
        while cp < cp_end {
            let fresh31 = cp;
            cp = cp.offset(1);
            row = *fresh31;
            (*Row.offset(row as isize)).shared2.mark = DEAD as int32_t;
        }
        pivot_row_length = pfree - pivot_row_start;
        if pivot_row_length > 0 as int32_t {
            pivot_row = *A.offset((*Col.offset(pivot_col as isize)).start as isize);
        } else {
            pivot_row = EMPTY as int32_t;
        }
        rp = &mut *A.offset(pivot_row_start as isize) as *mut int32_t;
        rp_end = rp.offset(pivot_row_length as isize);
        while rp < rp_end {
            let fresh32 = rp;
            rp = rp.offset(1);
            col = *fresh32;
            col_thickness = -(*Col.offset(col as isize)).shared1.thickness;
            (*Col.offset(col as isize)).shared1.thickness = col_thickness;
            cur_score = (*Col.offset(col as isize)).shared2.score;
            prev_col = (*Col.offset(col as isize)).shared3.prev;
            next_col = (*Col.offset(col as isize)).shared4.degree_next;
            if prev_col == EMPTY as int32_t {
                *head.offset(cur_score as isize) = next_col;
            } else {
                (*Col.offset(prev_col as isize)).shared4.degree_next = next_col;
            }
            if next_col != EMPTY as int32_t {
                (*Col.offset(next_col as isize)).shared3.prev = prev_col;
            }
            cp = &mut *A.offset((*Col.offset(col as isize)).start as isize) as *mut int32_t;
            cp_end = cp.offset((*Col.offset(col as isize)).length as isize);
            while cp < cp_end {
                let fresh33 = cp;
                cp = cp.offset(1);
                row = *fresh33;
                row_mark = (*Row.offset(row as isize)).shared2.mark;
                if row_mark < ALIVE as int32_t {
                    continue;
                }
                set_difference = row_mark - tag_mark;
                if set_difference < 0 as int32_t {
                    set_difference = (*Row.offset(row as isize)).shared1.degree;
                }
                set_difference -= col_thickness;
                if set_difference == 0 as int32_t && aggressive != 0 {
                    (*Row.offset(row as isize)).shared2.mark = DEAD as int32_t;
                } else {
                    (*Row.offset(row as isize)).shared2.mark = set_difference + tag_mark;
                }
            }
        }
        rp = &mut *A.offset(pivot_row_start as isize) as *mut int32_t;
        rp_end = rp.offset(pivot_row_length as isize);
        while rp < rp_end {
            let fresh34 = rp;
            rp = rp.offset(1);
            col = *fresh34;
            hash = 0 as uint32_t;
            cur_score = 0 as core::ffi::c_int as int32_t;
            cp = &mut *A.offset((*Col.offset(col as isize)).start as isize) as *mut int32_t;
            new_cp = cp;
            cp_end = cp.offset((*Col.offset(col as isize)).length as isize);
            while cp < cp_end {
                let fresh35 = cp;
                cp = cp.offset(1);
                row = *fresh35;
                row_mark = (*Row.offset(row as isize)).shared2.mark;
                if row_mark < ALIVE as int32_t {
                    continue;
                }
                let fresh36 = new_cp;
                new_cp = new_cp.offset(1);
                *fresh36 = row;
                hash = hash.wrapping_add(row as uint32_t);
                cur_score += row_mark - tag_mark;
                cur_score = if cur_score < n_col { cur_score } else { n_col };
            }
            (*Col.offset(col as isize)).length = new_cp.offset_from(
                &mut *A.offset((*Col.offset(col as isize)).start as isize) as *mut int32_t,
            ) as core::ffi::c_long as int32_t;
            if (*Col.offset(col as isize)).length == 0 as int32_t {
                (*Col.offset(col as isize)).start = DEAD_PRINCIPAL as int32_t;
                pivot_row_degree -= (*Col.offset(col as isize)).shared1.thickness;
                (*Col.offset(col as isize)).shared2.order = k;
                k += (*Col.offset(col as isize)).shared1.thickness;
            } else {
                (*Col.offset(col as isize)).shared2.score = cur_score;
                hash = hash.wrapping_rem((n_col + 1 as int32_t) as uint32_t);
                head_column = *head.offset(hash as isize);
                if head_column > EMPTY as int32_t {
                    first_col = (*Col.offset(head_column as isize)).shared3.headhash;
                    (*Col.offset(head_column as isize)).shared3.headhash = col;
                } else {
                    first_col = -(head_column + 2 as int32_t);
                    *head.offset(hash as isize) = -(col + 2 as int32_t);
                }
                (*Col.offset(col as isize)).shared4.hash_next = first_col;
                (*Col.offset(col as isize)).shared3.hash = hash as int32_t;
            }
        }
        detect_super_cols(Col, A, head, pivot_row_start, pivot_row_length);
        (*Col.offset(pivot_col as isize)).start = DEAD_PRINCIPAL as int32_t;
        tag_mark = clear_mark(tag_mark + max_deg + 1 as int32_t, max_mark, n_row, Row);
        rp = &mut *A.offset(pivot_row_start as isize) as *mut int32_t;
        new_rp = rp;
        rp_end = rp.offset(pivot_row_length as isize);
        while rp < rp_end {
            let fresh37 = rp;
            rp = rp.offset(1);
            col = *fresh37;
            if (*Col.offset(col as isize)).start < ALIVE as int32_t {
                continue;
            }
            let fresh38 = new_rp;
            new_rp = new_rp.offset(1);
            *fresh38 = col;
            *A.offset(
                ((*Col.offset(col as isize)).start + (*Col.offset(col as isize)).length) as isize,
            ) = pivot_row;
            (*Col.offset(col as isize)).length += 1;
            cur_score = (*Col.offset(col as isize)).shared2.score + pivot_row_degree;
            max_score = n_col - k - (*Col.offset(col as isize)).shared1.thickness;
            cur_score -= (*Col.offset(col as isize)).shared1.thickness;
            cur_score = if cur_score < max_score {
                cur_score
            } else {
                max_score
            };
            (*Col.offset(col as isize)).shared2.score = cur_score;
            next_col = *head.offset(cur_score as isize);
            (*Col.offset(col as isize)).shared4.degree_next = next_col;
            (*Col.offset(col as isize)).shared3.prev = EMPTY as int32_t;
            if next_col != EMPTY as int32_t {
                (*Col.offset(next_col as isize)).shared3.prev = col;
            }
            *head.offset(cur_score as isize) = col;
            min_score = if min_score < cur_score {
                min_score
            } else {
                cur_score
            };
        }
        if pivot_row_degree > 0 as int32_t {
            (*Row.offset(pivot_row as isize)).start = pivot_row_start;
            (*Row.offset(pivot_row as isize)).length =
                new_rp.offset_from(&mut *A.offset(pivot_row_start as isize) as *mut int32_t)
                    as core::ffi::c_long as int32_t;
            (*Row.offset(pivot_row as isize)).shared1.degree = pivot_row_degree;
            (*Row.offset(pivot_row as isize)).shared2.mark = 0 as core::ffi::c_int as int32_t;
        }
    }
    ngarbage
}

///  Order non-principal columns.
///
/// Before this, the [`find_ordering`] routine has ordered all of the principal columns (the
/// representatives of the supercolumns). The non-principal columns have not yet been ordered. This
/// routine orders those columns by walking up the parent tree (a column is a child of the column
/// which absorbed it). The final permutation vector is then placed in `p[0 ... n_col-1]`, with
/// `p[0]` being the first column, and `p[n_col-1]` being the last. It doesn't look like it at
/// first glance, but be assured that this routine takes time linear in the number of columns.
/// Although not immediately obvious, the time taken by this routine is `O(n_col)`, that is, linear
/// in the number of columns.
#[inline]
unsafe fn order_children(n_col: i32, col: &mut [Colamd_Col], p: &mut [i32]) {
    // === Order each non-principal column ==================================

    for i in 0..n_col {
        // Find an un-ordered non-principal column.
        debug_assert!(col[i as usize].start < ALIVE, "column is dead");
        if col[i as usize].start != DEAD_PRINCIPAL && col[i as usize].shared2.order == EMPTY {
            let mut parent = i;
            // Once found, find its principal parent.
            loop {
                parent = col[parent as usize].shared1.parent;
                if col[parent as usize].start == DEAD_PRINCIPAL {
                    break;
                }
            }
            // Now, order all un-ordered non-principal columns along path to this parent. Collapse
            // tree at the same time.
            let mut c = i;
            // Get order of parent.
            let mut order = col[parent as usize].shared2.order;
            loop {
                debug_assert_eq!(
                    col[c as usize].shared2.order, EMPTY,
                    "Pivot order is not set yet."
                );
                // Order this column.
                col[c as usize].shared2.order = order;
                order += 1;
                // Collapse tree.
                col[c as usize].shared1.parent = parent;

                // Get immediate parent of this column.
                c = col[c as usize].shared1.parent;

                // Continue until we hit an ordered column. There are guaranteed not to be any more
                // unordered columns above an ordered column.
                if col[c as usize].shared2.order != EMPTY {
                    break;
                }
            }

            // Re-order the super_col parent to largest order for this group.
            col[parent as usize].shared2.order = order;
        }
    }

    // === Generate the permutation =========================================

    for c in 0..n_col {
        p[col[c as usize].shared2.order as usize] = c;
    }
}
unsafe extern "C" fn detect_super_cols(
    Col: *mut Colamd_Col,
    A: *mut int32_t,
    head: *mut int32_t,
    row_start: int32_t,
    row_length: int32_t,
) {
    let mut hash: int32_t = 0;
    let mut rp: *mut int32_t = core::ptr::null_mut::<int32_t>();
    let mut c: int32_t = 0;
    let mut super_c: int32_t = 0;
    let mut cp1: *mut int32_t = core::ptr::null_mut::<int32_t>();
    let mut cp2: *mut int32_t = core::ptr::null_mut::<int32_t>();
    let mut length: int32_t = 0;
    let mut prev_c: int32_t = 0;
    let mut i: int32_t = 0;
    let mut rp_end: *mut int32_t = core::ptr::null_mut::<int32_t>();
    let mut head_column: int32_t = 0;
    let mut first_col: int32_t = 0;
    rp = &mut *A.offset(row_start as isize) as *mut int32_t;
    rp_end = rp.offset(row_length as isize);
    while rp < rp_end {
        let col = *rp;
        rp = rp.offset(1);
        if (*Col.offset(col as isize)).start < ALIVE as int32_t {
            continue;
        }
        hash = (*Col.offset(col as isize)).shared3.hash;
        head_column = *head.offset(hash as isize);
        if head_column > EMPTY as int32_t {
            first_col = (*Col.offset(head_column as isize)).shared3.headhash;
        } else {
            first_col = -(head_column + 2 as int32_t);
        }
        super_c = first_col;
        while super_c != EMPTY as int32_t {
            length = (*Col.offset(super_c as isize)).length;
            prev_c = super_c;
            c = (*Col.offset(super_c as isize)).shared4.hash_next;
            while c != EMPTY as int32_t {
                if (*Col.offset(c as isize)).length != length
                    || (*Col.offset(c as isize)).shared2.score
                        != (*Col.offset(super_c as isize)).shared2.score
                {
                    prev_c = c;
                } else {
                    cp1 = &mut *A.offset((*Col.offset(super_c as isize)).start as isize)
                        as *mut int32_t;
                    cp2 = &mut *A.offset((*Col.offset(c as isize)).start as isize) as *mut int32_t;
                    i = 0 as core::ffi::c_int as int32_t;
                    while i < length {
                        let fresh43 = cp1;
                        cp1 = cp1.offset(1);
                        let fresh44 = cp2;
                        cp2 = cp2.offset(1);
                        if *fresh43 != *fresh44 {
                            break;
                        }
                        i += 1;
                    }
                    if i != length {
                        prev_c = c;
                    } else {
                        (*Col.offset(super_c as isize)).shared1.thickness +=
                            (*Col.offset(c as isize)).shared1.thickness;
                        (*Col.offset(c as isize)).shared1.parent = super_c;
                        (*Col.offset(c as isize)).start = DEAD_NON_PRINCIPAL as int32_t;
                        (*Col.offset(c as isize)).shared2.order = EMPTY as int32_t;
                        (*Col.offset(prev_c as isize)).shared4.hash_next =
                            (*Col.offset(c as isize)).shared4.hash_next;
                    }
                }
                c = (*Col.offset(c as isize)).shared4.hash_next;
            }
            super_c = (*Col.offset(super_c as isize)).shared4.hash_next;
        }
        if head_column > EMPTY as int32_t {
            (*Col.offset(head_column as isize)).shared3.headhash = EMPTY as int32_t;
        } else {
            *head.offset(hash as isize) = EMPTY as int32_t;
        }
    }
}
unsafe extern "C" fn garbage_collection(
    n_row: int32_t,
    n_col: int32_t,
    Row: *mut Colamd_Row,
    Col: *mut Colamd_Col,
    A: *mut int32_t,
    pfree: *mut int32_t,
) -> int32_t {
    let mut psrc: *mut int32_t = core::ptr::null_mut::<int32_t>();
    let mut pdest: *mut int32_t = core::ptr::null_mut::<int32_t>();
    let mut j: int32_t = 0;
    let mut r: int32_t = 0;
    let mut c: int32_t = 0;
    let mut length: int32_t = 0;
    pdest = &mut *A.offset(0 as core::ffi::c_int as isize) as *mut int32_t;
    c = 0 as core::ffi::c_int as int32_t;
    while c < n_col {
        if (*Col.offset(c as isize)).start >= ALIVE as int32_t {
            psrc = &mut *A.offset((*Col.offset(c as isize)).start as isize) as *mut int32_t;
            (*Col.offset(c as isize)).start = pdest
                .offset_from(&mut *A.offset(0 as core::ffi::c_int as isize) as *mut int32_t)
                as core::ffi::c_long as int32_t;
            length = (*Col.offset(c as isize)).length;
            j = 0 as core::ffi::c_int as int32_t;
            while j < length {
                let fresh45 = psrc;
                psrc = psrc.offset(1);
                r = *fresh45;
                if (*Row.offset(r as isize)).shared2.mark >= ALIVE as int32_t {
                    let fresh46 = pdest;
                    pdest = pdest.offset(1);
                    *fresh46 = r;
                }
                j += 1;
            }
            (*Col.offset(c as isize)).length = pdest.offset_from(
                &mut *A.offset((*Col.offset(c as isize)).start as isize) as *mut int32_t,
            ) as core::ffi::c_long as int32_t;
        }
        c += 1;
    }
    r = 0 as core::ffi::c_int as int32_t;
    while r < n_row {
        if (*Row.offset(r as isize)).shared2.mark < ALIVE as int32_t
            || (*Row.offset(r as isize)).length == 0 as int32_t
        {
            (*Row.offset(r as isize)).shared2.mark = DEAD as int32_t;
        } else {
            psrc = &mut *A.offset((*Row.offset(r as isize)).start as isize) as *mut int32_t;
            (*Row.offset(r as isize)).shared2.first_column = *psrc;
            *psrc = -r - 1 as int32_t;
        }
        r += 1;
    }
    psrc = pdest;
    while psrc < pfree {
        let fresh47 = psrc;
        psrc = psrc.offset(1);
        if *fresh47 < 0 as int32_t {
            psrc = psrc.offset(-1);
            r = -*psrc - 1 as int32_t;
            *psrc = (*Row.offset(r as isize)).shared2.first_column;
            (*Row.offset(r as isize)).start = pdest
                .offset_from(&mut *A.offset(0 as core::ffi::c_int as isize) as *mut int32_t)
                as core::ffi::c_long as int32_t;
            length = (*Row.offset(r as isize)).length;
            j = 0 as core::ffi::c_int as int32_t;
            while j < length {
                let fresh48 = psrc;
                psrc = psrc.offset(1);
                c = *fresh48;
                if (*Col.offset(c as isize)).start >= ALIVE as int32_t {
                    let fresh49 = pdest;
                    pdest = pdest.offset(1);
                    *fresh49 = c;
                }
                j += 1;
            }
            (*Row.offset(r as isize)).length = pdest.offset_from(
                &mut *A.offset((*Row.offset(r as isize)).start as isize) as *mut int32_t,
            ) as core::ffi::c_long as int32_t;
        }
    }
    pdest.offset_from(&mut *A.offset(0 as core::ffi::c_int as isize) as *mut int32_t)
        as core::ffi::c_long as int32_t
}
unsafe extern "C" fn clear_mark(
    mut tag_mark: int32_t,
    max_mark: int32_t,
    n_row: int32_t,
    Row: *mut Colamd_Row,
) -> int32_t {
    let mut r: int32_t = 0;
    if tag_mark <= 0 as int32_t || tag_mark >= max_mark {
        r = 0 as core::ffi::c_int as int32_t;
        while r < n_row {
            if (*Row.offset(r as isize)).shared2.mark >= ALIVE as int32_t {
                (*Row.offset(r as isize)).shared2.mark = 0 as core::ffi::c_int as int32_t;
            }
            r += 1;
        }
        tag_mark = 1 as core::ffi::c_int as int32_t;
    }
    tag_mark
}
