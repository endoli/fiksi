// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This implements some hash-based collections backed by [`rustc_hash::FxBuildHasher`]. That
//! project states:
//!
//! "A speedy hash algorithm for use within rustc. The hashmap in liballoc by default uses
//! `SipHash` which isn’t quite as speedy as we want. In the compiler we’re not really worried
//! about DOS attempts, so we use a fast non-cryptographic hash."
//!
//! Here we have a similar use-case: our keys are very small and sequential. We're not too worried
//! about DOS attempts. Users can control which keys are masked out, but not the keys themselves.
//! We do we still want to get a reasonably good hash distribution in cases where there is a
//! pattern in how keys are masked out (meaning we can't just use "identity hashing").

use rustc_hash::FxBuildHasher;

/// Type-alias for a [`hashbrown::HashMap`] backed by a [`rustc_hash::FxHasher`].
///
/// See the [module-level](self) documentation for information about the hash function used.
pub(crate) type FxHashMap<K, V> = hashbrown::HashMap<K, V, FxBuildHasher>;

/// Type-alias for a [`hashbrown::HashSet`] backed by a [`rustc_hash::FxHasher`].
///
/// See the [module-level](self) documentation for information about the hash function used.
pub(crate) type FxHashSet<K> = hashbrown::HashSet<K, FxBuildHasher>;

/// Type-alias for a [`indexmap::IndexMap`] backed by a [`rustc_hash::FxHasher`].
///
/// See the [module-level](self) documentation for information about the hash function used.
pub(crate) type FxIndexMap<K, V> = indexmap::IndexMap<K, V, FxBuildHasher>;

/// Type-alias for a [`indexmap::IndexSet`] backed by a [`rustc_hash::FxHasher`].
///
/// See the [module-level](self) documentation for information about the hash function used.
pub(crate) type FxIndexSet<K> = indexmap::IndexSet<K, FxBuildHasher>;

/// An extension trait to be able to define some methods on type-aliased collections.
pub(crate) trait CollectionExt: Sized {
    fn new() -> Self {
        Self::with_capacity(0)
    }

    fn with_capacity(n: usize) -> Self;
}

impl<K, V> CollectionExt for FxHashMap<K, V> {
    fn with_capacity(n: usize) -> Self {
        Self::with_capacity_and_hasher(n, <_>::default())
    }
}

impl<K> CollectionExt for FxHashSet<K> {
    fn with_capacity(n: usize) -> Self {
        Self::with_capacity_and_hasher(n, <_>::default())
    }
}

impl<K, V> CollectionExt for FxIndexMap<K, V> {
    fn with_capacity(n: usize) -> Self {
        Self::with_capacity_and_hasher(n, <_>::default())
    }
}

impl<K> CollectionExt for FxIndexSet<K> {
    fn with_capacity(n: usize) -> Self {
        Self::with_capacity_and_hasher(n, <_>::default())
    }
}
