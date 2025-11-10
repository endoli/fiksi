// COLAMD, Copyright (c) 1998-2024, Timothy A. Davis and Stefan Larimore,
// All Rights Reserved.
// Copyright 2025 the Solvi Authors
// SPDX-License-Identifier: BSD-3-Clause

#![expect(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unused_assignments,
    unused_mut,
    unsafe_op_in_unsafe_fn,
    trivial_numeric_casts,
    clippy::assign_op_pattern,
    clippy::needless_return,
    clippy::nonminimal_bool,
    clippy::ptr_offset_with_cast,
    clippy::single_match,
    clippy::toplevel_ref_arg,
    clippy::unnecessary_unwrap,
    clippy::unnecessary_literal_unwrap,
    clippy::zero_ptr,
    reason = "transpiled using c2rust"
)]

unsafe extern "C" {
    fn sqrt(__x: core::ffi::c_double) -> core::ffi::c_double;
}

fn SuiteSparse_config_printf_func_get()
-> Option<unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int> {
    unimplemented!()
}

pub type size_t = usize;
pub type __int64_t = i64;
pub type __uint64_t = u64;
pub type int64_t = __int64_t;
pub type uint64_t = __uint64_t;
pub type Colamd_Row = Colamd_Row_struct;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Colamd_Row_struct {
    pub start: int64_t,
    pub length: int64_t,
    pub shared1: C2RustUnnamed_0,
    pub shared2: C2RustUnnamed,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed {
    pub mark: int64_t,
    pub first_column: int64_t,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed_0 {
    pub degree: int64_t,
    pub p: int64_t,
}
pub type Colamd_Col = Colamd_Col_struct;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Colamd_Col_struct {
    pub start: int64_t,
    pub length: int64_t,
    pub shared1: C2RustUnnamed_4,
    pub shared2: C2RustUnnamed_3,
    pub shared3: C2RustUnnamed_2,
    pub shared4: C2RustUnnamed_1,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed_1 {
    pub degree_next: int64_t,
    pub hash_next: int64_t,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed_2 {
    pub headhash: int64_t,
    pub hash: int64_t,
    pub prev: int64_t,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed_3 {
    pub score: int64_t,
    pub order: int64_t,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed_4 {
    pub thickness: int64_t,
    pub parent: int64_t,
}
pub const NULL: *mut core::ffi::c_void = 0 as *mut core::ffi::c_void;
pub const COLAMD_KNOBS: core::ffi::c_int = 20 as core::ffi::c_int;
pub const COLAMD_STATS: core::ffi::c_int = 20 as core::ffi::c_int;
pub const COLAMD_DENSE_ROW: core::ffi::c_int = 0 as core::ffi::c_int;
pub const COLAMD_DENSE_COL: core::ffi::c_int = 1 as core::ffi::c_int;
pub const COLAMD_AGGRESSIVE: core::ffi::c_int = 2 as core::ffi::c_int;
pub const COLAMD_DEFRAG_COUNT: core::ffi::c_int = 2 as core::ffi::c_int;
pub const COLAMD_STATUS: core::ffi::c_int = 3 as core::ffi::c_int;
pub const COLAMD_INFO1: core::ffi::c_int = 4 as core::ffi::c_int;
pub const COLAMD_INFO2: core::ffi::c_int = 5 as core::ffi::c_int;
pub const COLAMD_INFO3: core::ffi::c_int = 6 as core::ffi::c_int;
pub const COLAMD_OK: core::ffi::c_int = 0 as core::ffi::c_int;
pub const COLAMD_OK_BUT_JUMBLED: core::ffi::c_int = 1 as core::ffi::c_int;
pub const COLAMD_ERROR_A_not_present: core::ffi::c_int = -(1 as core::ffi::c_int);
pub const COLAMD_ERROR_p_not_present: core::ffi::c_int = -(2 as core::ffi::c_int);
pub const COLAMD_ERROR_nrow_negative: core::ffi::c_int = -(3 as core::ffi::c_int);
pub const COLAMD_ERROR_ncol_negative: core::ffi::c_int = -(4 as core::ffi::c_int);
pub const COLAMD_ERROR_nnz_negative: core::ffi::c_int = -(5 as core::ffi::c_int);
pub const COLAMD_ERROR_p0_nonzero: core::ffi::c_int = -(6 as core::ffi::c_int);
pub const COLAMD_ERROR_A_too_small: core::ffi::c_int = -(7 as core::ffi::c_int);
pub const COLAMD_ERROR_col_length_negative: core::ffi::c_int = -(8 as core::ffi::c_int);
pub const COLAMD_ERROR_row_index_out_of_bounds: core::ffi::c_int = -(9 as core::ffi::c_int);
pub const COLAMD_ERROR_out_of_memory: core::ffi::c_int = -(10 as core::ffi::c_int);
pub const TRUE: core::ffi::c_int = 1 as core::ffi::c_int;
pub const FALSE: core::ffi::c_int = 0 as core::ffi::c_int;
pub const EMPTY: core::ffi::c_int = -(1 as core::ffi::c_int);
pub const ALIVE: core::ffi::c_int = 0 as core::ffi::c_int;
pub const DEAD: core::ffi::c_int = -(1 as core::ffi::c_int);
pub const DEAD_PRINCIPAL: core::ffi::c_int = -(1 as core::ffi::c_int);
pub const DEAD_NON_PRINCIPAL: core::ffi::c_int = -(2 as core::ffi::c_int);
unsafe extern "C" fn t_add(mut a: size_t, mut b: size_t, mut ok: *mut core::ffi::c_int) -> size_t {
    *ok = (*ok != 0 && a.wrapping_add(b) >= (if a > b { a } else { b })) as core::ffi::c_int;
    return if *ok != 0 {
        a.wrapping_add(b)
    } else {
        0 as size_t
    };
}
unsafe extern "C" fn t_mult(mut a: size_t, mut k: size_t, mut ok: *mut core::ffi::c_int) -> size_t {
    let mut i: size_t = 0;
    let mut s: size_t = 0 as size_t;
    i = 0 as size_t;
    while i < k {
        s = t_add(s, a, ok);
        i = i.wrapping_add(1);
    }
    return s;
}

pub unsafe extern "C" fn colamd_l_recommended(
    mut nnz: int64_t,
    mut n_row: int64_t,
    mut n_col: int64_t,
) -> size_t {
    let mut s: size_t = 0;
    let mut c: size_t = 0;
    let mut r: size_t = 0;
    let mut ok: core::ffi::c_int = TRUE;
    if nnz < 0 as int64_t || n_row < 0 as int64_t || n_col < 0 as int64_t {
        return 0 as size_t;
    }
    s = t_mult(nnz as size_t, 2 as size_t, &mut ok);
    c = (t_mult(
        t_add(n_col as size_t, 1 as size_t, &mut ok),
        ::core::mem::size_of::<Colamd_Col>() as size_t,
        &mut ok,
    ))
    .wrapping_div(::core::mem::size_of::<int64_t>() as size_t);
    r = (t_mult(
        t_add(n_row as size_t, 1 as size_t, &mut ok),
        ::core::mem::size_of::<Colamd_Row>() as size_t,
        &mut ok,
    ))
    .wrapping_div(::core::mem::size_of::<int64_t>() as size_t);
    s = t_add(s, c, &mut ok);
    s = t_add(s, r, &mut ok);
    s = t_add(s, n_col as size_t, &mut ok);
    s = t_add(s, (nnz / 5 as int64_t) as size_t, &mut ok);
    return if ok != 0 { s } else { 0 as size_t };
}

pub unsafe extern "C" fn colamd_l_set_defaults(mut knobs: *mut core::ffi::c_double) {
    let mut i: int64_t = 0;
    if knobs.is_null() {
        return;
    }
    i = 0 as int64_t;
    while i < COLAMD_KNOBS as int64_t {
        *knobs.offset(i as isize) = 0 as core::ffi::c_int as core::ffi::c_double;
        i += 1;
    }
    *knobs.offset(COLAMD_DENSE_ROW as isize) = 10 as core::ffi::c_int as core::ffi::c_double;
    *knobs.offset(COLAMD_DENSE_COL as isize) = 10 as core::ffi::c_int as core::ffi::c_double;
    *knobs.offset(COLAMD_AGGRESSIVE as isize) = TRUE as core::ffi::c_double;
}

pub unsafe extern "C" fn symamd_l(
    mut n: int64_t,
    mut A: *mut int64_t,
    mut p: *mut int64_t,
    mut perm: *mut int64_t,
    mut knobs: *mut core::ffi::c_double,
    mut stats: *mut int64_t,
    mut allocate: Option<unsafe extern "C" fn(size_t, size_t) -> *mut core::ffi::c_void>,
    mut release: Option<unsafe extern "C" fn(*mut core::ffi::c_void) -> ()>,
) -> core::ffi::c_int {
    let mut count: *mut int64_t = 0 as *mut int64_t;
    let mut mark: *mut int64_t = 0 as *mut int64_t;
    let mut M: *mut int64_t = 0 as *mut int64_t;
    let mut Mlen: size_t = 0;
    let mut n_row: int64_t = 0;
    let mut nnz: int64_t = 0;
    let mut i: int64_t = 0;
    let mut j: int64_t = 0;
    let mut k: int64_t = 0;
    let mut mnz: int64_t = 0;
    let mut pp: int64_t = 0;
    let mut last_row: int64_t = 0;
    let mut length: int64_t = 0;
    let mut cknobs: [core::ffi::c_double; 20] = [0.; 20];
    let mut default_knobs: [core::ffi::c_double; 20] = [0.; 20];
    if stats.is_null() {
        return 0 as core::ffi::c_int;
    }
    i = 0 as int64_t;
    while i < COLAMD_STATS as int64_t {
        *stats.offset(i as isize) = 0 as int64_t;
        i += 1;
    }
    *stats.offset(COLAMD_STATUS as isize) = COLAMD_OK as int64_t;
    *stats.offset(COLAMD_INFO1 as isize) = -(1 as core::ffi::c_int) as int64_t;
    *stats.offset(COLAMD_INFO2 as isize) = -(1 as core::ffi::c_int) as int64_t;
    if A.is_null() {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_A_not_present as int64_t;
        return 0 as core::ffi::c_int;
    }
    if p.is_null() {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_p_not_present as int64_t;
        return 0 as core::ffi::c_int;
    }
    if n < 0 as int64_t {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_ncol_negative as int64_t;
        *stats.offset(COLAMD_INFO1 as isize) = n;
        return 0 as core::ffi::c_int;
    }
    nnz = *p.offset(n as isize);
    if nnz < 0 as int64_t {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_nnz_negative as int64_t;
        *stats.offset(COLAMD_INFO1 as isize) = nnz;
        return 0 as core::ffi::c_int;
    }
    if *p.offset(0 as core::ffi::c_int as isize) != 0 as int64_t {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_p0_nonzero as int64_t;
        *stats.offset(COLAMD_INFO1 as isize) = *p.offset(0 as core::ffi::c_int as isize);
        return 0 as core::ffi::c_int;
    }
    if knobs.is_null() {
        colamd_l_set_defaults(default_knobs.as_mut_ptr());
        knobs = default_knobs.as_mut_ptr() as *mut core::ffi::c_double;
    }
    count = (Some(allocate.expect("non-null function pointer"))).expect("non-null function pointer")(
        (n + 1 as int64_t) as size_t,
        ::core::mem::size_of::<int64_t>() as size_t,
    ) as *mut int64_t;
    if count.is_null() {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_out_of_memory as int64_t;
        return 0 as core::ffi::c_int;
    }
    mark = (Some(allocate.expect("non-null function pointer"))).expect("non-null function pointer")(
        (n + 1 as int64_t) as size_t,
        ::core::mem::size_of::<int64_t>() as size_t,
    ) as *mut int64_t;
    if mark.is_null() {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_out_of_memory as int64_t;
        (Some(release.expect("non-null function pointer"))).expect("non-null function pointer")(
            count as *mut core::ffi::c_void,
        );
        return 0 as core::ffi::c_int;
    }
    *stats.offset(COLAMD_INFO3 as isize) = 0 as int64_t;
    i = 0 as int64_t;
    while i < n {
        *mark.offset(i as isize) = -(1 as core::ffi::c_int) as int64_t;
        i += 1;
    }
    j = 0 as int64_t;
    while j < n {
        last_row = -(1 as core::ffi::c_int) as int64_t;
        length = *p.offset((j + 1 as int64_t) as isize) - *p.offset(j as isize);
        if length < 0 as int64_t {
            *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_col_length_negative as int64_t;
            *stats.offset(COLAMD_INFO1 as isize) = j;
            *stats.offset(COLAMD_INFO2 as isize) = length;
            (Some(release.expect("non-null function pointer"))).expect("non-null function pointer")(
                count as *mut core::ffi::c_void,
            );
            (Some(release.expect("non-null function pointer"))).expect("non-null function pointer")(
                mark as *mut core::ffi::c_void,
            );
            return 0 as core::ffi::c_int;
        }
        pp = *p.offset(j as isize);
        while pp < *p.offset((j + 1 as int64_t) as isize) {
            i = *A.offset(pp as isize);
            if i < 0 as int64_t || i >= n {
                *stats.offset(COLAMD_STATUS as isize) =
                    COLAMD_ERROR_row_index_out_of_bounds as int64_t;
                *stats.offset(COLAMD_INFO1 as isize) = j;
                *stats.offset(COLAMD_INFO2 as isize) = i;
                *stats.offset(COLAMD_INFO3 as isize) = n;
                (Some(release.expect("non-null function pointer")))
                    .expect("non-null function pointer")(
                    count as *mut core::ffi::c_void
                );
                (Some(release.expect("non-null function pointer")))
                    .expect("non-null function pointer")(
                    mark as *mut core::ffi::c_void
                );
                return 0 as core::ffi::c_int;
            }
            if i <= last_row || *mark.offset(i as isize) == j {
                *stats.offset(COLAMD_STATUS as isize) = COLAMD_OK_BUT_JUMBLED as int64_t;
                *stats.offset(COLAMD_INFO1 as isize) = j;
                *stats.offset(COLAMD_INFO2 as isize) = i;
                let ref mut fresh40 = *stats.offset(COLAMD_INFO3 as isize);
                *fresh40 += 1;
            }
            if i > j && *mark.offset(i as isize) != j {
                let ref mut fresh41 = *count.offset(i as isize);
                *fresh41 += 1;
                let ref mut fresh42 = *count.offset(j as isize);
                *fresh42 += 1;
            }
            *mark.offset(i as isize) = j;
            last_row = i;
            pp += 1;
        }
        j += 1;
    }
    *perm.offset(0 as core::ffi::c_int as isize) = 0 as int64_t;
    j = 1 as int64_t;
    while j <= n {
        *perm.offset(j as isize) =
            *perm.offset((j - 1 as int64_t) as isize) + *count.offset((j - 1 as int64_t) as isize);
        j += 1;
    }
    j = 0 as int64_t;
    while j < n {
        *count.offset(j as isize) = *perm.offset(j as isize);
        j += 1;
    }
    mnz = *perm.offset(n as isize);
    n_row = mnz / 2 as int64_t;
    Mlen = colamd_l_recommended(mnz, n_row, n);
    M = (Some(allocate.expect("non-null function pointer"))).expect("non-null function pointer")(
        Mlen,
        ::core::mem::size_of::<int64_t>() as size_t,
    ) as *mut int64_t;
    if M.is_null() {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_out_of_memory as int64_t;
        (Some(release.expect("non-null function pointer"))).expect("non-null function pointer")(
            count as *mut core::ffi::c_void,
        );
        (Some(release.expect("non-null function pointer"))).expect("non-null function pointer")(
            mark as *mut core::ffi::c_void,
        );
        return 0 as core::ffi::c_int;
    }
    k = 0 as int64_t;
    if *stats.offset(COLAMD_STATUS as isize) == COLAMD_OK as int64_t {
        j = 0 as int64_t;
        while j < n {
            pp = *p.offset(j as isize);
            while pp < *p.offset((j + 1 as int64_t) as isize) {
                i = *A.offset(pp as isize);
                if i > j {
                    let ref mut fresh43 = *count.offset(i as isize);
                    let fresh44 = *fresh43;
                    *fresh43 = *fresh43 + 1;
                    *M.offset(fresh44 as isize) = k;
                    let ref mut fresh45 = *count.offset(j as isize);
                    let fresh46 = *fresh45;
                    *fresh45 = *fresh45 + 1;
                    *M.offset(fresh46 as isize) = k;
                    k += 1;
                }
                pp += 1;
            }
            j += 1;
        }
    } else {
        i = 0 as int64_t;
        while i < n {
            *mark.offset(i as isize) = -(1 as core::ffi::c_int) as int64_t;
            i += 1;
        }
        j = 0 as int64_t;
        while j < n {
            pp = *p.offset(j as isize);
            while pp < *p.offset((j + 1 as int64_t) as isize) {
                i = *A.offset(pp as isize);
                if i > j && *mark.offset(i as isize) != j {
                    let ref mut fresh47 = *count.offset(i as isize);
                    let fresh48 = *fresh47;
                    *fresh47 = *fresh47 + 1;
                    *M.offset(fresh48 as isize) = k;
                    let ref mut fresh49 = *count.offset(j as isize);
                    let fresh50 = *fresh49;
                    *fresh49 = *fresh49 + 1;
                    *M.offset(fresh50 as isize) = k;
                    k += 1;
                    *mark.offset(i as isize) = j;
                }
                pp += 1;
            }
            j += 1;
        }
    }
    (Some(release.expect("non-null function pointer"))).expect("non-null function pointer")(
        count as *mut core::ffi::c_void,
    );
    (Some(release.expect("non-null function pointer"))).expect("non-null function pointer")(
        mark as *mut core::ffi::c_void,
    );
    i = 0 as int64_t;
    while i < COLAMD_KNOBS as int64_t {
        cknobs[i as usize] = *knobs.offset(i as isize);
        i += 1;
    }
    cknobs[COLAMD_DENSE_ROW as usize] = -(1 as core::ffi::c_int) as core::ffi::c_double;
    cknobs[COLAMD_DENSE_COL as usize] = *knobs.offset(COLAMD_DENSE_ROW as isize);
    colamd_l(
        n_row,
        n,
        Mlen as int64_t,
        M as *mut int64_t,
        perm,
        cknobs.as_mut_ptr(),
        stats,
    );
    *stats.offset(COLAMD_DENSE_ROW as isize) = *stats.offset(COLAMD_DENSE_COL as isize);
    (Some(release.expect("non-null function pointer"))).expect("non-null function pointer")(
        M as *mut core::ffi::c_void,
    );
    return 1 as core::ffi::c_int;
}

pub unsafe extern "C" fn colamd_l(
    mut n_row: int64_t,
    mut n_col: int64_t,
    mut Alen: int64_t,
    mut A: *mut int64_t,
    mut p: *mut int64_t,
    mut knobs: *mut core::ffi::c_double,
    mut stats: *mut int64_t,
) -> core::ffi::c_int {
    let mut i: int64_t = 0;
    let mut nnz: int64_t = 0;
    let mut Row_size: size_t = 0;
    let mut Col_size: size_t = 0;
    let mut need: size_t = 0;
    let mut Row: *mut Colamd_Row = 0 as *mut Colamd_Row;
    let mut Col: *mut Colamd_Col = 0 as *mut Colamd_Col;
    let mut n_col2: int64_t = 0;
    let mut n_row2: int64_t = 0;
    let mut ngarbage: int64_t = 0;
    let mut max_deg: int64_t = 0;
    let mut default_knobs: [core::ffi::c_double; 20] = [0.; 20];
    let mut aggressive: int64_t = 0;
    let mut ok: core::ffi::c_int = 0;
    if stats.is_null() {
        return 0 as core::ffi::c_int;
    }
    i = 0 as int64_t;
    while i < COLAMD_STATS as int64_t {
        *stats.offset(i as isize) = 0 as int64_t;
        i += 1;
    }
    *stats.offset(COLAMD_STATUS as isize) = COLAMD_OK as int64_t;
    *stats.offset(COLAMD_INFO1 as isize) = -(1 as core::ffi::c_int) as int64_t;
    *stats.offset(COLAMD_INFO2 as isize) = -(1 as core::ffi::c_int) as int64_t;
    if A.is_null() {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_A_not_present as int64_t;
        return 0 as core::ffi::c_int;
    }
    if p.is_null() {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_p_not_present as int64_t;
        return 0 as core::ffi::c_int;
    }
    if n_row < 0 as int64_t {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_nrow_negative as int64_t;
        *stats.offset(COLAMD_INFO1 as isize) = n_row;
        return 0 as core::ffi::c_int;
    }
    if n_col < 0 as int64_t {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_ncol_negative as int64_t;
        *stats.offset(COLAMD_INFO1 as isize) = n_col;
        return 0 as core::ffi::c_int;
    }
    nnz = *p.offset(n_col as isize);
    if nnz < 0 as int64_t {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_nnz_negative as int64_t;
        *stats.offset(COLAMD_INFO1 as isize) = nnz;
        return 0 as core::ffi::c_int;
    }
    if *p.offset(0 as core::ffi::c_int as isize) != 0 as int64_t {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_p0_nonzero as int64_t;
        *stats.offset(COLAMD_INFO1 as isize) = *p.offset(0 as core::ffi::c_int as isize);
        return 0 as core::ffi::c_int;
    }
    if knobs.is_null() {
        colamd_l_set_defaults(default_knobs.as_mut_ptr());
        knobs = default_knobs.as_mut_ptr() as *mut core::ffi::c_double;
    }
    aggressive = (*knobs.offset(COLAMD_AGGRESSIVE as isize) != FALSE as core::ffi::c_double)
        as core::ffi::c_int as int64_t;
    ok = TRUE;
    Col_size = (t_mult(
        t_add(n_col as size_t, 1 as size_t, &mut ok),
        ::core::mem::size_of::<Colamd_Col>() as size_t,
        &mut ok,
    ))
    .wrapping_div(::core::mem::size_of::<int64_t>() as size_t);
    Row_size = (t_mult(
        t_add(n_row as size_t, 1 as size_t, &mut ok),
        ::core::mem::size_of::<Colamd_Row>() as size_t,
        &mut ok,
    ))
    .wrapping_div(::core::mem::size_of::<int64_t>() as size_t);
    need = t_mult(nnz as size_t, 2 as size_t, &mut ok);
    need = t_add(need, n_col as size_t, &mut ok);
    need = t_add(need, Col_size, &mut ok);
    need = t_add(need, Row_size, &mut ok);
    if ok == 0 || need > Alen as size_t {
        *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_A_too_small as int64_t;
        *stats.offset(COLAMD_INFO1 as isize) = need as int64_t;
        *stats.offset(COLAMD_INFO2 as isize) = Alen;
        return 0 as core::ffi::c_int;
    }
    Alen = (Alen as core::ffi::c_ulong)
        .wrapping_sub(Col_size.wrapping_add(Row_size) as core::ffi::c_ulong) as int64_t
        as int64_t;
    Col = &mut *A.offset(Alen as isize) as *mut int64_t as *mut Colamd_Col;
    Row = &mut *A.offset((Alen as size_t).wrapping_add(Col_size) as isize) as *mut int64_t
        as *mut Colamd_Row;
    if init_rows_cols(
        n_row,
        n_col,
        Row as *mut Colamd_Row,
        Col as *mut Colamd_Col,
        A,
        p,
        stats,
    ) == 0
    {
        return 0 as core::ffi::c_int;
    }
    init_scoring(
        n_row,
        n_col,
        Row as *mut Colamd_Row,
        Col as *mut Colamd_Col,
        A,
        p,
        knobs,
        &mut n_row2,
        &mut n_col2,
        &mut max_deg,
    );
    ngarbage = find_ordering(
        n_row,
        n_col,
        Alen,
        Row as *mut Colamd_Row,
        Col as *mut Colamd_Col,
        A,
        p,
        n_col2,
        max_deg,
        2 as int64_t * nnz,
        aggressive,
    );
    order_children(n_col, Col as *mut Colamd_Col, p);
    *stats.offset(COLAMD_DENSE_ROW as isize) = n_row - n_row2;
    *stats.offset(COLAMD_DENSE_COL as isize) = n_col - n_col2;
    *stats.offset(COLAMD_DEFRAG_COUNT as isize) = ngarbage;
    return 1 as core::ffi::c_int;
}

pub unsafe extern "C" fn colamd_l_report(mut stats: *mut int64_t) {
    print_report(
        b"colamd\0" as *const u8 as *const core::ffi::c_char as *mut core::ffi::c_char,
        stats,
    );
}

pub unsafe extern "C" fn symamd_l_report(mut stats: *mut int64_t) {
    print_report(
        b"symamd\0" as *const u8 as *const core::ffi::c_char as *mut core::ffi::c_char,
        stats,
    );
}
unsafe extern "C" fn init_rows_cols(
    mut n_row: int64_t,
    mut n_col: int64_t,
    mut Row: *mut Colamd_Row,
    mut Col: *mut Colamd_Col,
    mut A: *mut int64_t,
    mut p: *mut int64_t,
    mut stats: *mut int64_t,
) -> int64_t {
    let mut col: int64_t = 0;
    let mut row: int64_t = 0;
    let mut cp: *mut int64_t = 0 as *mut int64_t;
    let mut cp_end: *mut int64_t = 0 as *mut int64_t;
    let mut rp: *mut int64_t = 0 as *mut int64_t;
    let mut rp_end: *mut int64_t = 0 as *mut int64_t;
    let mut last_row: int64_t = 0;
    col = 0 as int64_t;
    while col < n_col {
        (*Col.offset(col as isize)).start = *p.offset(col as isize);
        (*Col.offset(col as isize)).length =
            *p.offset((col + 1 as int64_t) as isize) - *p.offset(col as isize);
        if (*Col.offset(col as isize)).length < 0 as int64_t {
            *stats.offset(COLAMD_STATUS as isize) = COLAMD_ERROR_col_length_negative as int64_t;
            *stats.offset(COLAMD_INFO1 as isize) = col;
            *stats.offset(COLAMD_INFO2 as isize) = (*Col.offset(col as isize)).length;
            return 0 as int64_t;
        }
        (*Col.offset(col as isize)).shared1.thickness = 1 as int64_t;
        (*Col.offset(col as isize)).shared2.score = 0 as int64_t;
        (*Col.offset(col as isize)).shared3.prev = EMPTY as int64_t;
        (*Col.offset(col as isize)).shared4.degree_next = EMPTY as int64_t;
        col += 1;
    }
    *stats.offset(COLAMD_INFO3 as isize) = 0 as int64_t;
    row = 0 as int64_t;
    while row < n_row {
        (*Row.offset(row as isize)).length = 0 as int64_t;
        (*Row.offset(row as isize)).shared2.mark = -(1 as core::ffi::c_int) as int64_t;
        row += 1;
    }
    col = 0 as int64_t;
    while col < n_col {
        last_row = -(1 as core::ffi::c_int) as int64_t;
        cp = &mut *A.offset(*p.offset(col as isize) as isize) as *mut int64_t;
        cp_end = &mut *A.offset(*p.offset((col + 1 as int64_t) as isize) as isize) as *mut int64_t;
        while cp < cp_end {
            let fresh27 = cp;
            cp = cp.offset(1);
            row = *fresh27;
            if row < 0 as int64_t || row >= n_row {
                *stats.offset(COLAMD_STATUS as isize) =
                    COLAMD_ERROR_row_index_out_of_bounds as int64_t;
                *stats.offset(COLAMD_INFO1 as isize) = col;
                *stats.offset(COLAMD_INFO2 as isize) = row;
                *stats.offset(COLAMD_INFO3 as isize) = n_row;
                return 0 as int64_t;
            }
            if row <= last_row || (*Row.offset(row as isize)).shared2.mark == col {
                *stats.offset(COLAMD_STATUS as isize) = COLAMD_OK_BUT_JUMBLED as int64_t;
                *stats.offset(COLAMD_INFO1 as isize) = col;
                *stats.offset(COLAMD_INFO2 as isize) = row;
                let ref mut fresh28 = *stats.offset(COLAMD_INFO3 as isize);
                *fresh28 += 1;
            }
            if (*Row.offset(row as isize)).shared2.mark != col {
                let ref mut fresh29 = (*Row.offset(row as isize)).length;
                *fresh29 += 1;
            } else {
                let ref mut fresh30 = (*Col.offset(col as isize)).length;
                *fresh30 -= 1;
            }
            (*Row.offset(row as isize)).shared2.mark = col;
            last_row = row;
        }
        col += 1;
    }
    (*Row.offset(0 as core::ffi::c_int as isize)).start = *p.offset(n_col as isize);
    (*Row.offset(0 as core::ffi::c_int as isize)).shared1.p =
        (*Row.offset(0 as core::ffi::c_int as isize)).start;
    (*Row.offset(0 as core::ffi::c_int as isize)).shared2.mark =
        -(1 as core::ffi::c_int) as int64_t;
    row = 1 as int64_t;
    while row < n_row {
        (*Row.offset(row as isize)).start = (*Row.offset((row - 1 as int64_t) as isize)).start
            + (*Row.offset((row - 1 as int64_t) as isize)).length;
        (*Row.offset(row as isize)).shared1.p = (*Row.offset(row as isize)).start;
        (*Row.offset(row as isize)).shared2.mark = -(1 as core::ffi::c_int) as int64_t;
        row += 1;
    }
    if *stats.offset(COLAMD_STATUS as isize) == COLAMD_OK_BUT_JUMBLED as int64_t {
        col = 0 as int64_t;
        while col < n_col {
            cp = &mut *A.offset(*p.offset(col as isize) as isize) as *mut int64_t;
            cp_end =
                &mut *A.offset(*p.offset((col + 1 as int64_t) as isize) as isize) as *mut int64_t;
            while cp < cp_end {
                let fresh31 = cp;
                cp = cp.offset(1);
                row = *fresh31;
                if (*Row.offset(row as isize)).shared2.mark != col {
                    let ref mut fresh32 = (*Row.offset(row as isize)).shared1.p;
                    let fresh33 = *fresh32;
                    *fresh32 = *fresh32 + 1;
                    *A.offset(fresh33 as isize) = col;
                    (*Row.offset(row as isize)).shared2.mark = col;
                }
            }
            col += 1;
        }
    } else {
        col = 0 as int64_t;
        while col < n_col {
            cp = &mut *A.offset(*p.offset(col as isize) as isize) as *mut int64_t;
            cp_end =
                &mut *A.offset(*p.offset((col + 1 as int64_t) as isize) as isize) as *mut int64_t;
            while cp < cp_end {
                let fresh34 = cp;
                cp = cp.offset(1);
                let ref mut fresh35 = (*Row.offset(*fresh34 as isize)).shared1.p;
                let fresh36 = *fresh35;
                *fresh35 = *fresh35 + 1;
                *A.offset(fresh36 as isize) = col;
            }
            col += 1;
        }
    }
    row = 0 as int64_t;
    while row < n_row {
        (*Row.offset(row as isize)).shared2.mark = 0 as int64_t;
        (*Row.offset(row as isize)).shared1.degree = (*Row.offset(row as isize)).length;
        row += 1;
    }
    if *stats.offset(COLAMD_STATUS as isize) == COLAMD_OK_BUT_JUMBLED as int64_t {
        (*Col.offset(0 as core::ffi::c_int as isize)).start = 0 as int64_t;
        *p.offset(0 as core::ffi::c_int as isize) =
            (*Col.offset(0 as core::ffi::c_int as isize)).start;
        col = 1 as int64_t;
        while col < n_col {
            (*Col.offset(col as isize)).start = (*Col.offset((col - 1 as int64_t) as isize)).start
                + (*Col.offset((col - 1 as int64_t) as isize)).length;
            *p.offset(col as isize) = (*Col.offset(col as isize)).start;
            col += 1;
        }
        row = 0 as int64_t;
        while row < n_row {
            rp = &mut *A.offset((*Row.offset(row as isize)).start as isize) as *mut int64_t;
            rp_end = rp.offset((*Row.offset(row as isize)).length as isize);
            while rp < rp_end {
                let fresh37 = rp;
                rp = rp.offset(1);
                let ref mut fresh38 = *p.offset(*fresh37 as isize);
                let fresh39 = *fresh38;
                *fresh38 = *fresh38 + 1;
                *A.offset(fresh39 as isize) = row;
            }
            row += 1;
        }
    }
    return 1 as int64_t;
}
unsafe extern "C" fn init_scoring(
    mut n_row: int64_t,
    mut n_col: int64_t,
    mut Row: *mut Colamd_Row,
    mut Col: *mut Colamd_Col,
    mut A: *mut int64_t,
    mut head: *mut int64_t,
    mut knobs: *mut core::ffi::c_double,
    mut p_n_row2: *mut int64_t,
    mut p_n_col2: *mut int64_t,
    mut p_max_deg: *mut int64_t,
) {
    let mut c: int64_t = 0;
    let mut r: int64_t = 0;
    let mut row: int64_t = 0;
    let mut cp: *mut int64_t = 0 as *mut int64_t;
    let mut deg: int64_t = 0;
    let mut cp_end: *mut int64_t = 0 as *mut int64_t;
    let mut new_cp: *mut int64_t = 0 as *mut int64_t;
    let mut col_length: int64_t = 0;
    let mut score: int64_t = 0;
    let mut n_col2: int64_t = 0;
    let mut n_row2: int64_t = 0;
    let mut dense_row_count: int64_t = 0;
    let mut dense_col_count: int64_t = 0;
    let mut min_score: int64_t = 0;
    let mut max_deg: int64_t = 0;
    let mut next_col: int64_t = 0;
    if *knobs.offset(COLAMD_DENSE_ROW as isize) < 0 as core::ffi::c_int as core::ffi::c_double {
        dense_row_count = n_col - 1 as int64_t;
    } else {
        dense_row_count = (if 16.0f64
            > *knobs.offset(0 as core::ffi::c_int as isize) * sqrt(n_col as core::ffi::c_double)
        {
            16.0f64
        } else {
            *knobs.offset(0 as core::ffi::c_int as isize) * sqrt(n_col as core::ffi::c_double)
        }) as int64_t;
    }
    if *knobs.offset(COLAMD_DENSE_COL as isize) < 0 as core::ffi::c_int as core::ffi::c_double {
        dense_col_count = n_row - 1 as int64_t;
    } else {
        dense_col_count = (if 16.0f64
            > *knobs.offset(1 as core::ffi::c_int as isize)
                * sqrt((if n_row < n_col { n_row } else { n_col }) as core::ffi::c_double)
        {
            16.0f64
        } else {
            *knobs.offset(1 as core::ffi::c_int as isize)
                * sqrt((if n_row < n_col { n_row } else { n_col }) as core::ffi::c_double)
        }) as int64_t;
    }
    max_deg = 0 as int64_t;
    n_col2 = n_col;
    n_row2 = n_row;
    c = n_col - 1 as int64_t;
    while c >= 0 as int64_t {
        deg = (*Col.offset(c as isize)).length;
        if deg == 0 as int64_t {
            n_col2 -= 1;
            (*Col.offset(c as isize)).shared2.order = n_col2;
            (*Col.offset(c as isize)).start = DEAD_PRINCIPAL as int64_t;
        }
        c -= 1;
    }
    c = n_col - 1 as int64_t;
    while c >= 0 as int64_t {
        if !((*Col.offset(c as isize)).start < ALIVE as int64_t) {
            deg = (*Col.offset(c as isize)).length;
            if deg > dense_col_count {
                n_col2 -= 1;
                (*Col.offset(c as isize)).shared2.order = n_col2;
                cp = &mut *A.offset((*Col.offset(c as isize)).start as isize) as *mut int64_t;
                cp_end = cp.offset((*Col.offset(c as isize)).length as isize);
                while cp < cp_end {
                    let fresh23 = cp;
                    cp = cp.offset(1);
                    let ref mut fresh24 = (*Row.offset(*fresh23 as isize)).shared1.degree;
                    *fresh24 -= 1;
                }
                (*Col.offset(c as isize)).start = DEAD_PRINCIPAL as int64_t;
            }
        }
        c -= 1;
    }
    r = 0 as int64_t;
    while r < n_row {
        deg = (*Row.offset(r as isize)).shared1.degree;
        if deg > dense_row_count || deg == 0 as int64_t {
            (*Row.offset(r as isize)).shared2.mark = DEAD as int64_t;
            n_row2 -= 1;
        } else {
            max_deg = if max_deg > deg { max_deg } else { deg };
        }
        r += 1;
    }
    c = n_col - 1 as int64_t;
    while c >= 0 as int64_t {
        if !((*Col.offset(c as isize)).start < ALIVE as int64_t) {
            score = 0 as int64_t;
            cp = &mut *A.offset((*Col.offset(c as isize)).start as isize) as *mut int64_t;
            new_cp = cp;
            cp_end = cp.offset((*Col.offset(c as isize)).length as isize);
            while cp < cp_end {
                let fresh25 = cp;
                cp = cp.offset(1);
                row = *fresh25;
                if (*Row.offset(row as isize)).shared2.mark < ALIVE as int64_t {
                    continue;
                }
                let fresh26 = new_cp;
                new_cp = new_cp.offset(1);
                *fresh26 = row;
                score = (score as core::ffi::c_long
                    + ((*Row.offset(row as isize)).shared1.degree - 1 as int64_t)
                        as core::ffi::c_long) as int64_t;
                score = if score < n_col { score } else { n_col };
            }
            col_length = new_cp.offset_from(
                &mut *A.offset((*Col.offset(c as isize)).start as isize) as *mut int64_t,
            ) as core::ffi::c_long as int64_t;
            if col_length == 0 as int64_t {
                n_col2 -= 1;
                (*Col.offset(c as isize)).shared2.order = n_col2;
                (*Col.offset(c as isize)).start = DEAD_PRINCIPAL as int64_t;
            } else {
                (*Col.offset(c as isize)).length = col_length;
                (*Col.offset(c as isize)).shared2.score = score;
            }
        }
        c -= 1;
    }
    c = 0 as int64_t;
    while c <= n_col {
        *head.offset(c as isize) = EMPTY as int64_t;
        c += 1;
    }
    min_score = n_col;
    c = n_col - 1 as int64_t;
    while c >= 0 as int64_t {
        if (*Col.offset(c as isize)).start >= ALIVE as int64_t {
            score = (*Col.offset(c as isize)).shared2.score;
            next_col = *head.offset(score as isize);
            (*Col.offset(c as isize)).shared3.prev = EMPTY as int64_t;
            (*Col.offset(c as isize)).shared4.degree_next = next_col;
            if next_col != EMPTY as int64_t {
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
    mut n_row: int64_t,
    mut n_col: int64_t,
    mut Alen: int64_t,
    mut Row: *mut Colamd_Row,
    mut Col: *mut Colamd_Col,
    mut A: *mut int64_t,
    mut head: *mut int64_t,
    mut n_col2: int64_t,
    mut max_deg: int64_t,
    mut pfree: int64_t,
    mut aggressive: int64_t,
) -> int64_t {
    let mut k: int64_t = 0;
    let mut pivot_col: int64_t = 0;
    let mut cp: *mut int64_t = 0 as *mut int64_t;
    let mut rp: *mut int64_t = 0 as *mut int64_t;
    let mut pivot_row: int64_t = 0;
    let mut new_cp: *mut int64_t = 0 as *mut int64_t;
    let mut new_rp: *mut int64_t = 0 as *mut int64_t;
    let mut pivot_row_start: int64_t = 0;
    let mut pivot_row_degree: int64_t = 0;
    let mut pivot_row_length: int64_t = 0;
    let mut pivot_col_score: int64_t = 0;
    let mut needed_memory: int64_t = 0;
    let mut cp_end: *mut int64_t = 0 as *mut int64_t;
    let mut rp_end: *mut int64_t = 0 as *mut int64_t;
    let mut row: int64_t = 0;
    let mut col: int64_t = 0;
    let mut max_score: int64_t = 0;
    let mut cur_score: int64_t = 0;
    let mut hash: uint64_t = 0;
    let mut head_column: int64_t = 0;
    let mut first_col: int64_t = 0;
    let mut tag_mark: int64_t = 0;
    let mut row_mark: int64_t = 0;
    let mut set_difference: int64_t = 0;
    let mut min_score: int64_t = 0;
    let mut col_thickness: int64_t = 0;
    let mut max_mark: int64_t = 0;
    let mut pivot_col_thickness: int64_t = 0;
    let mut prev_col: int64_t = 0;
    let mut next_col: int64_t = 0;
    let mut ngarbage: int64_t = 0;
    max_mark = INT_MAX as int64_t - n_col;
    tag_mark = clear_mark(0 as int64_t, max_mark, n_row, Row);
    min_score = 0 as int64_t;
    ngarbage = 0 as int64_t;
    k = 0 as int64_t;
    while k < n_col2 {
        while *head.offset(min_score as isize) == EMPTY as int64_t && min_score < n_col {
            min_score += 1;
        }
        pivot_col = *head.offset(min_score as isize);
        next_col = (*Col.offset(pivot_col as isize)).shared4.degree_next;
        *head.offset(min_score as isize) = next_col;
        if next_col != EMPTY as int64_t {
            (*Col.offset(next_col as isize)).shared3.prev = EMPTY as int64_t;
        }
        pivot_col_score = (*Col.offset(pivot_col as isize)).shared2.score;
        (*Col.offset(pivot_col as isize)).shared2.order = k;
        pivot_col_thickness = (*Col.offset(pivot_col as isize)).shared1.thickness;
        k = (k as core::ffi::c_long + pivot_col_thickness as core::ffi::c_long) as int64_t;
        needed_memory = if pivot_col_score < n_col - k {
            pivot_col_score
        } else {
            n_col - k
        };
        if pfree + needed_memory >= Alen {
            pfree = garbage_collection(n_row, n_col, Row, Col, A, &mut *A.offset(pfree as isize));
            ngarbage += 1;
            tag_mark = clear_mark(0 as int64_t, max_mark, n_row, Row);
        }
        pivot_row_start = pfree;
        pivot_row_degree = 0 as int64_t;
        (*Col.offset(pivot_col as isize)).shared1.thickness = -pivot_col_thickness;
        cp = &mut *A.offset((*Col.offset(pivot_col as isize)).start as isize) as *mut int64_t;
        cp_end = cp.offset((*Col.offset(pivot_col as isize)).length as isize);
        while cp < cp_end {
            let fresh1 = cp;
            cp = cp.offset(1);
            row = *fresh1;
            if (*Row.offset(row as isize)).shared2.mark >= ALIVE as int64_t {
                rp = &mut *A.offset((*Row.offset(row as isize)).start as isize) as *mut int64_t;
                rp_end = rp.offset((*Row.offset(row as isize)).length as isize);
                while rp < rp_end {
                    let fresh2 = rp;
                    rp = rp.offset(1);
                    col = *fresh2;
                    col_thickness = (*Col.offset(col as isize)).shared1.thickness;
                    if col_thickness > 0 as int64_t
                        && (*Col.offset(col as isize)).start >= ALIVE as int64_t
                    {
                        (*Col.offset(col as isize)).shared1.thickness = -col_thickness;
                        let fresh3 = pfree;
                        pfree = pfree + 1;
                        *A.offset(fresh3 as isize) = col;
                        pivot_row_degree = (pivot_row_degree as core::ffi::c_long
                            + col_thickness as core::ffi::c_long)
                            as int64_t;
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
        cp = &mut *A.offset((*Col.offset(pivot_col as isize)).start as isize) as *mut int64_t;
        cp_end = cp.offset((*Col.offset(pivot_col as isize)).length as isize);
        while cp < cp_end {
            let fresh4 = cp;
            cp = cp.offset(1);
            row = *fresh4;
            (*Row.offset(row as isize)).shared2.mark = DEAD as int64_t;
        }
        pivot_row_length = pfree - pivot_row_start;
        if pivot_row_length > 0 as int64_t {
            pivot_row = *A.offset((*Col.offset(pivot_col as isize)).start as isize);
        } else {
            pivot_row = EMPTY as int64_t;
        }
        rp = &mut *A.offset(pivot_row_start as isize) as *mut int64_t;
        rp_end = rp.offset(pivot_row_length as isize);
        while rp < rp_end {
            let fresh5 = rp;
            rp = rp.offset(1);
            col = *fresh5;
            col_thickness = -(*Col.offset(col as isize)).shared1.thickness;
            (*Col.offset(col as isize)).shared1.thickness = col_thickness;
            cur_score = (*Col.offset(col as isize)).shared2.score;
            prev_col = (*Col.offset(col as isize)).shared3.prev;
            next_col = (*Col.offset(col as isize)).shared4.degree_next;
            if prev_col == EMPTY as int64_t {
                *head.offset(cur_score as isize) = next_col;
            } else {
                (*Col.offset(prev_col as isize)).shared4.degree_next = next_col;
            }
            if next_col != EMPTY as int64_t {
                (*Col.offset(next_col as isize)).shared3.prev = prev_col;
            }
            cp = &mut *A.offset((*Col.offset(col as isize)).start as isize) as *mut int64_t;
            cp_end = cp.offset((*Col.offset(col as isize)).length as isize);
            while cp < cp_end {
                let fresh6 = cp;
                cp = cp.offset(1);
                row = *fresh6;
                row_mark = (*Row.offset(row as isize)).shared2.mark;
                if row_mark < ALIVE as int64_t {
                    continue;
                }
                set_difference = row_mark - tag_mark;
                if set_difference < 0 as int64_t {
                    set_difference = (*Row.offset(row as isize)).shared1.degree;
                }
                set_difference = (set_difference as core::ffi::c_long
                    - col_thickness as core::ffi::c_long)
                    as int64_t;
                if set_difference == 0 as int64_t && aggressive != 0 {
                    (*Row.offset(row as isize)).shared2.mark = DEAD as int64_t;
                } else {
                    (*Row.offset(row as isize)).shared2.mark = set_difference + tag_mark;
                }
            }
        }
        rp = &mut *A.offset(pivot_row_start as isize) as *mut int64_t;
        rp_end = rp.offset(pivot_row_length as isize);
        while rp < rp_end {
            let fresh7 = rp;
            rp = rp.offset(1);
            col = *fresh7;
            hash = 0 as uint64_t;
            cur_score = 0 as int64_t;
            cp = &mut *A.offset((*Col.offset(col as isize)).start as isize) as *mut int64_t;
            new_cp = cp;
            cp_end = cp.offset((*Col.offset(col as isize)).length as isize);
            while cp < cp_end {
                let fresh8 = cp;
                cp = cp.offset(1);
                row = *fresh8;
                row_mark = (*Row.offset(row as isize)).shared2.mark;
                if row_mark < ALIVE as int64_t {
                    continue;
                }
                let fresh9 = new_cp;
                new_cp = new_cp.offset(1);
                *fresh9 = row;
                hash = (hash as core::ffi::c_ulong).wrapping_add(row as core::ffi::c_ulong)
                    as uint64_t as uint64_t;
                cur_score = (cur_score as core::ffi::c_long
                    + (row_mark - tag_mark) as core::ffi::c_long)
                    as int64_t;
                cur_score = if cur_score < n_col { cur_score } else { n_col };
            }
            (*Col.offset(col as isize)).length = new_cp.offset_from(
                &mut *A.offset((*Col.offset(col as isize)).start as isize) as *mut int64_t,
            ) as core::ffi::c_long as int64_t;
            if (*Col.offset(col as isize)).length == 0 as int64_t {
                (*Col.offset(col as isize)).start = DEAD_PRINCIPAL as int64_t;
                pivot_row_degree = (pivot_row_degree as core::ffi::c_long
                    - (*Col.offset(col as isize)).shared1.thickness as core::ffi::c_long)
                    as int64_t;
                (*Col.offset(col as isize)).shared2.order = k;
                k = (k as core::ffi::c_long
                    + (*Col.offset(col as isize)).shared1.thickness as core::ffi::c_long)
                    as int64_t;
            } else {
                (*Col.offset(col as isize)).shared2.score = cur_score;
                hash = (hash as core::ffi::c_ulong)
                    .wrapping_rem((n_col + 1 as int64_t) as core::ffi::c_ulong)
                    as uint64_t as uint64_t;
                head_column = *head.offset(hash as isize);
                if head_column > EMPTY as int64_t {
                    first_col = (*Col.offset(head_column as isize)).shared3.headhash;
                    (*Col.offset(head_column as isize)).shared3.headhash = col;
                } else {
                    first_col = -(head_column + 2 as int64_t);
                    *head.offset(hash as isize) = -(col + 2 as int64_t);
                }
                (*Col.offset(col as isize)).shared4.hash_next = first_col;
                (*Col.offset(col as isize)).shared3.hash = hash as int64_t;
            }
        }
        detect_super_cols(Col, A, head, pivot_row_start, pivot_row_length);
        (*Col.offset(pivot_col as isize)).start = DEAD_PRINCIPAL as int64_t;
        tag_mark = clear_mark(tag_mark + max_deg + 1 as int64_t, max_mark, n_row, Row);
        rp = &mut *A.offset(pivot_row_start as isize) as *mut int64_t;
        new_rp = rp;
        rp_end = rp.offset(pivot_row_length as isize);
        while rp < rp_end {
            let fresh10 = rp;
            rp = rp.offset(1);
            col = *fresh10;
            if (*Col.offset(col as isize)).start < ALIVE as int64_t {
                continue;
            }
            let fresh11 = new_rp;
            new_rp = new_rp.offset(1);
            *fresh11 = col;
            let ref mut fresh12 = (*Col.offset(col as isize)).length;
            let fresh13 = *fresh12;
            *fresh12 = *fresh12 + 1;
            *A.offset(((*Col.offset(col as isize)).start + fresh13) as isize) = pivot_row;
            cur_score = (*Col.offset(col as isize)).shared2.score + pivot_row_degree;
            max_score = n_col - k - (*Col.offset(col as isize)).shared1.thickness;
            cur_score = (cur_score as core::ffi::c_long
                - (*Col.offset(col as isize)).shared1.thickness as core::ffi::c_long)
                as int64_t;
            cur_score = if cur_score < max_score {
                cur_score
            } else {
                max_score
            };
            (*Col.offset(col as isize)).shared2.score = cur_score;
            next_col = *head.offset(cur_score as isize);
            (*Col.offset(col as isize)).shared4.degree_next = next_col;
            (*Col.offset(col as isize)).shared3.prev = EMPTY as int64_t;
            if next_col != EMPTY as int64_t {
                (*Col.offset(next_col as isize)).shared3.prev = col;
            }
            *head.offset(cur_score as isize) = col;
            min_score = if min_score < cur_score {
                min_score
            } else {
                cur_score
            };
        }
        if pivot_row_degree > 0 as int64_t {
            (*Row.offset(pivot_row as isize)).start = pivot_row_start;
            (*Row.offset(pivot_row as isize)).length =
                new_rp.offset_from(&mut *A.offset(pivot_row_start as isize) as *mut int64_t)
                    as core::ffi::c_long as int64_t;
            (*Row.offset(pivot_row as isize)).shared1.degree = pivot_row_degree;
            (*Row.offset(pivot_row as isize)).shared2.mark = 0 as int64_t;
        }
    }
    return ngarbage;
}
unsafe extern "C" fn order_children(
    mut n_col: int64_t,
    mut Col: *mut Colamd_Col,
    mut p: *mut int64_t,
) {
    let mut i: int64_t = 0;
    let mut c: int64_t = 0;
    let mut parent: int64_t = 0;
    let mut order: int64_t = 0;
    i = 0 as int64_t;
    while i < n_col {
        if !((*Col.offset(i as isize)).start == DEAD_PRINCIPAL as int64_t)
            && (*Col.offset(i as isize)).shared2.order == EMPTY as int64_t
        {
            parent = i;
            loop {
                parent = (*Col.offset(parent as isize)).shared1.parent;
                if (*Col.offset(parent as isize)).start == DEAD_PRINCIPAL as int64_t {
                    break;
                }
            }
            c = i;
            order = (*Col.offset(parent as isize)).shared2.order;
            loop {
                let fresh0 = order;
                order = order + 1;
                (*Col.offset(c as isize)).shared2.order = fresh0;
                (*Col.offset(c as isize)).shared1.parent = parent;
                c = (*Col.offset(c as isize)).shared1.parent;
                if !((*Col.offset(c as isize)).shared2.order == EMPTY as int64_t) {
                    break;
                }
            }
            (*Col.offset(parent as isize)).shared2.order = order;
        }
        i += 1;
    }
    c = 0 as int64_t;
    while c < n_col {
        *p.offset((*Col.offset(c as isize)).shared2.order as isize) = c;
        c += 1;
    }
}
unsafe extern "C" fn detect_super_cols(
    mut Col: *mut Colamd_Col,
    mut A: *mut int64_t,
    mut head: *mut int64_t,
    mut row_start: int64_t,
    mut row_length: int64_t,
) {
    let mut hash: int64_t = 0;
    let mut rp: *mut int64_t = 0 as *mut int64_t;
    let mut c: int64_t = 0;
    let mut super_c: int64_t = 0;
    let mut cp1: *mut int64_t = 0 as *mut int64_t;
    let mut cp2: *mut int64_t = 0 as *mut int64_t;
    let mut length: int64_t = 0;
    let mut prev_c: int64_t = 0;
    let mut i: int64_t = 0;
    let mut rp_end: *mut int64_t = 0 as *mut int64_t;
    let mut col: int64_t = 0;
    let mut head_column: int64_t = 0;
    let mut first_col: int64_t = 0;
    rp = &mut *A.offset(row_start as isize) as *mut int64_t;
    rp_end = rp.offset(row_length as isize);
    while rp < rp_end {
        let fresh14 = rp;
        rp = rp.offset(1);
        col = *fresh14;
        if (*Col.offset(col as isize)).start < ALIVE as int64_t {
            continue;
        }
        hash = (*Col.offset(col as isize)).shared3.hash;
        head_column = *head.offset(hash as isize);
        if head_column > EMPTY as int64_t {
            first_col = (*Col.offset(head_column as isize)).shared3.headhash;
        } else {
            first_col = -(head_column + 2 as int64_t);
        }
        super_c = first_col;
        while super_c != EMPTY as int64_t {
            length = (*Col.offset(super_c as isize)).length;
            prev_c = super_c;
            c = (*Col.offset(super_c as isize)).shared4.hash_next;
            while c != EMPTY as int64_t {
                if (*Col.offset(c as isize)).length != length
                    || (*Col.offset(c as isize)).shared2.score
                        != (*Col.offset(super_c as isize)).shared2.score
                {
                    prev_c = c;
                } else {
                    cp1 = &mut *A.offset((*Col.offset(super_c as isize)).start as isize)
                        as *mut int64_t;
                    cp2 = &mut *A.offset((*Col.offset(c as isize)).start as isize) as *mut int64_t;
                    i = 0 as int64_t;
                    while i < length {
                        let fresh15 = cp1;
                        cp1 = cp1.offset(1);
                        let fresh16 = cp2;
                        cp2 = cp2.offset(1);
                        if *fresh15 != *fresh16 {
                            break;
                        }
                        i += 1;
                    }
                    if i != length {
                        prev_c = c;
                    } else {
                        let ref mut fresh17 = (*Col.offset(super_c as isize)).shared1.thickness;
                        *fresh17 = (*fresh17 as core::ffi::c_long
                            + (*Col.offset(c as isize)).shared1.thickness as core::ffi::c_long)
                            as int64_t;
                        (*Col.offset(c as isize)).shared1.parent = super_c;
                        (*Col.offset(c as isize)).start = DEAD_NON_PRINCIPAL as int64_t;
                        (*Col.offset(c as isize)).shared2.order = EMPTY as int64_t;
                        (*Col.offset(prev_c as isize)).shared4.hash_next =
                            (*Col.offset(c as isize)).shared4.hash_next;
                    }
                }
                c = (*Col.offset(c as isize)).shared4.hash_next;
            }
            super_c = (*Col.offset(super_c as isize)).shared4.hash_next;
        }
        if head_column > EMPTY as int64_t {
            (*Col.offset(head_column as isize)).shared3.headhash = EMPTY as int64_t;
        } else {
            *head.offset(hash as isize) = EMPTY as int64_t;
        }
    }
}
unsafe extern "C" fn garbage_collection(
    mut n_row: int64_t,
    mut n_col: int64_t,
    mut Row: *mut Colamd_Row,
    mut Col: *mut Colamd_Col,
    mut A: *mut int64_t,
    mut pfree: *mut int64_t,
) -> int64_t {
    let mut psrc: *mut int64_t = 0 as *mut int64_t;
    let mut pdest: *mut int64_t = 0 as *mut int64_t;
    let mut j: int64_t = 0;
    let mut r: int64_t = 0;
    let mut c: int64_t = 0;
    let mut length: int64_t = 0;
    pdest = &mut *A.offset(0 as core::ffi::c_int as isize) as *mut int64_t;
    c = 0 as int64_t;
    while c < n_col {
        if (*Col.offset(c as isize)).start >= ALIVE as int64_t {
            psrc = &mut *A.offset((*Col.offset(c as isize)).start as isize) as *mut int64_t;
            (*Col.offset(c as isize)).start = pdest
                .offset_from(&mut *A.offset(0 as core::ffi::c_int as isize) as *mut int64_t)
                as core::ffi::c_long as int64_t;
            length = (*Col.offset(c as isize)).length;
            j = 0 as int64_t;
            while j < length {
                let fresh18 = psrc;
                psrc = psrc.offset(1);
                r = *fresh18;
                if (*Row.offset(r as isize)).shared2.mark >= ALIVE as int64_t {
                    let fresh19 = pdest;
                    pdest = pdest.offset(1);
                    *fresh19 = r;
                }
                j += 1;
            }
            (*Col.offset(c as isize)).length = pdest.offset_from(
                &mut *A.offset((*Col.offset(c as isize)).start as isize) as *mut int64_t,
            ) as core::ffi::c_long as int64_t;
        }
        c += 1;
    }
    r = 0 as int64_t;
    while r < n_row {
        if (*Row.offset(r as isize)).shared2.mark < ALIVE as int64_t
            || (*Row.offset(r as isize)).length == 0 as int64_t
        {
            (*Row.offset(r as isize)).shared2.mark = DEAD as int64_t;
        } else {
            psrc = &mut *A.offset((*Row.offset(r as isize)).start as isize) as *mut int64_t;
            (*Row.offset(r as isize)).shared2.first_column = *psrc;
            *psrc = -r - 1 as int64_t;
        }
        r += 1;
    }
    psrc = pdest;
    while psrc < pfree {
        let fresh20 = psrc;
        psrc = psrc.offset(1);
        if *fresh20 < 0 as int64_t {
            psrc = psrc.offset(-1);
            r = -*psrc - 1 as int64_t;
            *psrc = (*Row.offset(r as isize)).shared2.first_column;
            (*Row.offset(r as isize)).start = pdest
                .offset_from(&mut *A.offset(0 as core::ffi::c_int as isize) as *mut int64_t)
                as core::ffi::c_long as int64_t;
            length = (*Row.offset(r as isize)).length;
            j = 0 as int64_t;
            while j < length {
                let fresh21 = psrc;
                psrc = psrc.offset(1);
                c = *fresh21;
                if (*Col.offset(c as isize)).start >= ALIVE as int64_t {
                    let fresh22 = pdest;
                    pdest = pdest.offset(1);
                    *fresh22 = c;
                }
                j += 1;
            }
            (*Row.offset(r as isize)).length = pdest.offset_from(
                &mut *A.offset((*Row.offset(r as isize)).start as isize) as *mut int64_t,
            ) as core::ffi::c_long as int64_t;
        }
    }
    return pdest.offset_from(&mut *A.offset(0 as core::ffi::c_int as isize) as *mut int64_t)
        as core::ffi::c_long as int64_t;
}
unsafe extern "C" fn clear_mark(
    mut tag_mark: int64_t,
    mut max_mark: int64_t,
    mut n_row: int64_t,
    mut Row: *mut Colamd_Row,
) -> int64_t {
    let mut r: int64_t = 0;
    if tag_mark <= 0 as int64_t || tag_mark >= max_mark {
        r = 0 as int64_t;
        while r < n_row {
            if (*Row.offset(r as isize)).shared2.mark >= ALIVE as int64_t {
                (*Row.offset(r as isize)).shared2.mark = 0 as int64_t;
            }
            r += 1;
        }
        tag_mark = 1 as int64_t;
    }
    return tag_mark;
}
unsafe extern "C" fn print_report(mut method: *mut core::ffi::c_char, mut stats: *mut int64_t) {
    let mut i1: int64_t = 0;
    let mut i2: int64_t = 0;
    let mut i3: int64_t = 0;
    let mut printf_func: Option<
        unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
    > = None;
    printf_func = SuiteSparse_config_printf_func_get();
    if printf_func.is_some() {
        printf_func.expect("non-null function pointer")(
            b"\n%s version %d.%d.%d, %s: \0" as *const u8 as *const core::ffi::c_char,
            method,
            3 as core::ffi::c_int,
            3 as core::ffi::c_int,
            5 as core::ffi::c_int,
            b"July 25, 2025\0" as *const u8 as *const core::ffi::c_char,
        );
    }
    if stats.is_null() {
        let mut printf_func_0: Option<
            unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
        > = None;
        printf_func_0 = SuiteSparse_config_printf_func_get();
        if printf_func_0.is_some() {
            printf_func_0.expect("non-null function pointer")(
                b"No statistics available.\n\0" as *const u8 as *const core::ffi::c_char,
            );
        }
        return;
    }
    i1 = *stats.offset(COLAMD_INFO1 as isize);
    i2 = *stats.offset(COLAMD_INFO2 as isize);
    i3 = *stats.offset(COLAMD_INFO3 as isize);
    if *stats.offset(COLAMD_STATUS as isize) >= 0 as int64_t {
        let mut printf_func_1: Option<
            unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
        > = None;
        printf_func_1 = SuiteSparse_config_printf_func_get();
        if printf_func_1.is_some() {
            printf_func_1.expect("non-null function pointer")(
                b"OK.  \0" as *const u8 as *const core::ffi::c_char,
            );
        }
    } else {
        let mut printf_func_2: Option<
            unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
        > = None;
        printf_func_2 = SuiteSparse_config_printf_func_get();
        if printf_func_2.is_some() {
            printf_func_2.expect("non-null function pointer")(
                b"ERROR.  \0" as *const u8 as *const core::ffi::c_char,
            );
        }
    }
    let mut current_block_146: u64;
    match *stats.offset(COLAMD_STATUS as isize) {
        1 => {
            let mut printf_func_3: Option<
                unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
            > = None;
            printf_func_3 = SuiteSparse_config_printf_func_get();
            if printf_func_3.is_some() {
                printf_func_3.expect("non-null function pointer")(
                    b"Matrix has unsorted or duplicate row indices.\n\0" as *const u8
                        as *const core::ffi::c_char,
                );
            }
            let mut printf_func_4: Option<
                unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
            > = None;
            printf_func_4 = SuiteSparse_config_printf_func_get();
            if printf_func_4.is_some() {
                printf_func_4.expect("non-null function pointer")(
                    b"%s: number of duplicate or out-of-order row indices: %d\n\0" as *const u8
                        as *const core::ffi::c_char,
                    method,
                    i3,
                );
            }
            let mut printf_func_5: Option<
                unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
            > = None;
            printf_func_5 = SuiteSparse_config_printf_func_get();
            if printf_func_5.is_some() {
                printf_func_5.expect("non-null function pointer")(
                    b"%s: last seen duplicate or out-of-order row index:   %d\n\0" as *const u8
                        as *const core::ffi::c_char,
                    method,
                    i2,
                );
            }
            let mut printf_func_6: Option<
                unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
            > = None;
            printf_func_6 = SuiteSparse_config_printf_func_get();
            if printf_func_6.is_some() {
                printf_func_6.expect("non-null function pointer")(
                    b"%s: last seen in column:                             %d\0" as *const u8
                        as *const core::ffi::c_char,
                    method,
                    i1,
                );
            }
            current_block_146 = 12591474248834837570;
        }
        0 => {
            current_block_146 = 12591474248834837570;
        }
        -1 => {
            let mut printf_func_11: Option<
                unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
            > = None;
            printf_func_11 = SuiteSparse_config_printf_func_get();
            if printf_func_11.is_some() {
                printf_func_11.expect("non-null function pointer")(
                    b"Array A (row indices of matrix) not present.\n\0" as *const u8
                        as *const core::ffi::c_char,
                );
            }
            current_block_146 = 7337917895049117968;
        }
        -2 => {
            let mut printf_func_12: Option<
                unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
            > = None;
            printf_func_12 = SuiteSparse_config_printf_func_get();
            if printf_func_12.is_some() {
                printf_func_12.expect("non-null function pointer")(
                    b"Array p (column pointers for matrix) not present.\n\0" as *const u8
                        as *const core::ffi::c_char,
                );
            }
            current_block_146 = 7337917895049117968;
        }
        -3 => {
            let mut printf_func_13: Option<
                unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
            > = None;
            printf_func_13 = SuiteSparse_config_printf_func_get();
            if printf_func_13.is_some() {
                printf_func_13.expect("non-null function pointer")(
                    b"Invalid number of rows (%d).\n\0" as *const u8 as *const core::ffi::c_char,
                    i1,
                );
            }
            current_block_146 = 7337917895049117968;
        }
        -4 => {
            let mut printf_func_14: Option<
                unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
            > = None;
            printf_func_14 = SuiteSparse_config_printf_func_get();
            if printf_func_14.is_some() {
                printf_func_14.expect("non-null function pointer")(
                    b"Invalid number of columns (%d).\n\0" as *const u8 as *const core::ffi::c_char,
                    i1,
                );
            }
            current_block_146 = 7337917895049117968;
        }
        -5 => {
            let mut printf_func_15: Option<
                unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
            > = None;
            printf_func_15 = SuiteSparse_config_printf_func_get();
            if printf_func_15.is_some() {
                printf_func_15.expect("non-null function pointer")(
                    b"Invalid number of nonzero entries (%d).\n\0" as *const u8
                        as *const core::ffi::c_char,
                    i1,
                );
            }
            current_block_146 = 7337917895049117968;
        }
        -6 => {
            let mut printf_func_16: Option<
                unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
            > = None;
            printf_func_16 = SuiteSparse_config_printf_func_get();
            if printf_func_16.is_some() {
                printf_func_16.expect("non-null function pointer")(
                    b"Invalid column pointer, p [0] = %d, must be zero.\n\0" as *const u8
                        as *const core::ffi::c_char,
                    i1,
                );
            }
            current_block_146 = 7337917895049117968;
        }
        -7 => {
            let mut printf_func_17: Option<
                unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
            > = None;
            printf_func_17 = SuiteSparse_config_printf_func_get();
            if printf_func_17.is_some() {
                printf_func_17.expect("non-null function pointer")(
                    b"Array A too small.\n\0" as *const u8 as *const core::ffi::c_char,
                );
            }
            let mut printf_func_18: Option<
                unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
            > = None;
            printf_func_18 = SuiteSparse_config_printf_func_get();
            if printf_func_18.is_some() {
                printf_func_18.expect("non-null function pointer")(
                    b"        Need Alen >= %d, but given only Alen = %d.\n\0" as *const u8
                        as *const core::ffi::c_char,
                    i1,
                    i2,
                );
            }
            current_block_146 = 7337917895049117968;
        }
        -8 => {
            let mut printf_func_19: Option<
                unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
            > = None;
            printf_func_19 = SuiteSparse_config_printf_func_get();
            if printf_func_19.is_some() {
                printf_func_19.expect("non-null function pointer")(
                    b"Column %d has a negative number of nonzero entries (%d).\n\0" as *const u8
                        as *const core::ffi::c_char,
                    i1,
                    i2,
                );
            }
            current_block_146 = 7337917895049117968;
        }
        -9 => {
            let mut printf_func_20: Option<
                unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
            > = None;
            printf_func_20 = SuiteSparse_config_printf_func_get();
            if printf_func_20.is_some() {
                printf_func_20.expect("non-null function pointer")(
                    b"Row index (row %d) out of bounds (%d to %d) in column %d.\n\0" as *const u8
                        as *const core::ffi::c_char,
                    i2,
                    0 as core::ffi::c_int,
                    i3 - 1 as int64_t,
                    i1,
                );
            }
            current_block_146 = 7337917895049117968;
        }
        -10 => {
            let mut printf_func_21: Option<
                unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
            > = None;
            printf_func_21 = SuiteSparse_config_printf_func_get();
            if printf_func_21.is_some() {
                printf_func_21.expect("non-null function pointer")(
                    b"Out of memory.\n\0" as *const u8 as *const core::ffi::c_char,
                );
            }
            current_block_146 = 7337917895049117968;
        }
        _ => {
            current_block_146 = 7337917895049117968;
        }
    }
    match current_block_146 {
        12591474248834837570 => {
            let mut printf_func_7: Option<
                unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
            > = None;
            printf_func_7 = SuiteSparse_config_printf_func_get();
            if printf_func_7.is_some() {
                printf_func_7.expect("non-null function pointer")(
                    b"\n\0" as *const u8 as *const core::ffi::c_char,
                );
            }
            let mut printf_func_8: Option<
                unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
            > = None;
            printf_func_8 = SuiteSparse_config_printf_func_get();
            if printf_func_8.is_some() {
                printf_func_8.expect("non-null function pointer")(
                    b"%s: number of dense or empty rows ignored:           %d\n\0" as *const u8
                        as *const core::ffi::c_char,
                    method,
                    *stats.offset(0 as core::ffi::c_int as isize),
                );
            }
            let mut printf_func_9: Option<
                unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
            > = None;
            printf_func_9 = SuiteSparse_config_printf_func_get();
            if printf_func_9.is_some() {
                printf_func_9.expect("non-null function pointer")(
                    b"%s: number of dense or empty columns ignored:        %d\n\0" as *const u8
                        as *const core::ffi::c_char,
                    method,
                    *stats.offset(1 as core::ffi::c_int as isize),
                );
            }
            let mut printf_func_10: Option<
                unsafe extern "C" fn(*const core::ffi::c_char, ...) -> core::ffi::c_int,
            > = None;
            printf_func_10 = SuiteSparse_config_printf_func_get();
            if printf_func_10.is_some() {
                printf_func_10.expect("non-null function pointer")(
                    b"%s: number of garbage collections performed:         %d\n\0" as *const u8
                        as *const core::ffi::c_char,
                    method,
                    *stats.offset(2 as core::ffi::c_int as isize),
                );
            }
        }
        _ => {}
    };
}
pub const INT_MAX: core::ffi::c_int = __INT_MAX__;
pub const __INT_MAX__: core::ffi::c_int = 2147483647 as core::ffi::c_int;
