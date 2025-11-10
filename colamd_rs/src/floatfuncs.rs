// COLAMD, Copyright (c) 1998-2024, Timothy A. Davis and Stefan Larimore,
// All Rights Reserved.
// Copyright 2025 the Solvi Authors
// SPDX-License-Identifier: BSD-3-Clause

//! Shims for math functions that ordinarily come from std.

/// Defines a trait that chooses between libstd or libm implementations of float methods.
macro_rules! define_float_funcs {
    ($(
        fn $name:ident(self $(,$arg:ident: $arg_ty:ty)*) -> $ret:ty
            => $lname:ident/$lfname:ident;
    )+) => {

        /// Since core doesn't depend upon libm, this provides libm implementations
        /// of float functions which are typically provided by the std library, when
        /// the `std` feature is not enabled.
        ///
        /// For documentation see the respective functions in the std library.
        #[cfg(not(feature = "std"))]
        #[expect(dead_code, reason = "Tasteful YAGNI.")]
        pub(crate) trait FloatFuncs : Sized {
            $(fn $name(self $(,$arg: $arg_ty)*) -> $ret;)+
        }

        #[cfg(not(feature = "std"))]
        impl FloatFuncs for f32 {
            $(fn $name(self $(,$arg: $arg_ty)*) -> $ret {
                #[cfg(feature = "libm")]
                return libm::$lfname(self $(,$arg)*);

                #[cfg(not(feature = "libm"))]
                compile_error!("`colamd_rs` requires either the `std` or `libm` feature")
            })+
        }

        #[cfg(not(feature = "std"))]
        impl FloatFuncs for f64 {
            $(fn $name(self $(,$arg: $arg_ty)*) -> $ret {
                #[cfg(feature = "libm")]
                return libm::$lname(self $(,$arg)*);

                #[cfg(not(feature = "libm"))]
                compile_error!("`colamd_rs` requires either the `std` or `libm` feature")
            })+
        }
    }
}

define_float_funcs! {
    fn sqrt(self) -> Self => sqrt/sqrtf;
}
