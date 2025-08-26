// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This implements some hash-based collections backed by [`hashbrown::DefaultHashBuilder`], which
//! (as of this writing) is a re-export for [`foldhash::fast::RandomState`][foldhash]. The
//! `foldhash` project states:
//!
//! "This crate provides foldhash, a fast, non-cryptographic, minimally DoS-resistant hashing
//! algorithm designed for computational uses such as hashmaps, bloom filters, count sketching,
//! etc."
//!
//! In our use-case, our keys are very small and sequential. We're not too worried about DOS
//! attempts. Users can control which keys are masked out, but not the keys themselves. We do we
//! still want to get a reasonably good hash distribution in cases where there is a pattern in how
//! keys are masked out (meaning we can't just use "identity hashing").
//!
//! For hashing small numeric keys, foldhash optimizes to just a few instructions.
//!
//! [foldhash]: <https://docs.rs/foldhash/0.1.2/foldhash/fast/struct.RandomState.html>

/// Type-alias for a [`indexmap::IndexMap`] backed by a [`hashbrown::DefaultHashBuilder`].
///
/// See the [module-level](self) documentation for information about the hash function used.
pub(crate) type IndexMap<K, V> = indexmap::IndexMap<K, V, hashbrown::DefaultHashBuilder>;

/// Type-alias for a [`indexmap::IndexSet`] backed by a [`hashbrown::DefaultHashBuilder`].
///
/// See the [module-level](self) documentation for information about the hash function used.
pub(crate) type IndexSet<K> = indexmap::IndexSet<K, hashbrown::DefaultHashBuilder>;

/// An extension trait to be able to define some methods on type-aliased collections.
pub(crate) trait CollectionExt: Sized {
    fn new() -> Self {
        Self::with_capacity(0)
    }

    fn with_capacity(n: usize) -> Self;
}

impl<K, V> CollectionExt for IndexMap<K, V> {
    fn with_capacity(n: usize) -> Self {
        Self::with_capacity_and_hasher(n, <_>::default())
    }
}

impl<K> CollectionExt for IndexSet<K> {
    fn with_capacity(n: usize) -> Self {
        Self::with_capacity_and_hasher(n, <_>::default())
    }
}
