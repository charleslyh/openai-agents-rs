//! Typed business data attached to a run (Rust-only; Python carries one `TContext` object).
//!
//! The SDK is not generic over the user context. Instead a run carries an [`Extensions`] map
//! with at most one value per Rust type, so independent extensions can each bring their own
//! dependency (a database handle, a user id) without agreeing on one application-wide struct.
//!
//! ```
//! use openai_agents_core::Extensions;
//!
//! struct Db(&'static str);
//! struct UserId(u64);
//!
//! let extensions = Extensions::new().with(Db("main")).with(UserId(7));
//! assert_eq!(extensions.get::<UserId>().map(|u| u.0), Some(7));
//! assert!(extensions.get::<String>().is_none());
//! ```

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

/// A shared, type-erased value stored in [`Extensions`].
pub type ContextValue = Arc<dyn Any + Send + Sync>;

#[derive(Clone)]
struct Slot {
    value: ContextValue,
    /// Known when the value was inserted by type; `None` for an already erased value.
    type_name: Option<&'static str>,
}

/// A map from Rust type to one shared value of that type.
///
/// Cloning is cheap: values are behind `Arc`. Inserting a second value of the same type
/// replaces the first (and logs a warning in debug builds, where it is almost always a bug).
#[derive(Clone, Default)]
pub struct Extensions {
    slots: HashMap<TypeId, Slot>,
}

impl Extensions {
    /// An empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store `value`, replacing any earlier value of the same type.
    pub fn insert<T: Any + Send + Sync>(&mut self, value: T) {
        self.put(
            TypeId::of::<T>(),
            Slot {
                value: Arc::new(value),
                type_name: Some(std::any::type_name::<T>()),
            },
        );
    }

    /// Builder form of [`Extensions::insert`].
    #[must_use]
    pub fn with<T: Any + Send + Sync>(mut self, value: T) -> Self {
        self.insert(value);
        self
    }

    /// Store an already shared value under the type it holds.
    ///
    /// This is how the single-value `RunOptions::context` is carried: the value lands in the
    /// slot of its own concrete type.
    pub fn insert_shared(&mut self, value: ContextValue) {
        let type_id = (*value).type_id();
        self.put(
            type_id,
            Slot {
                value,
                type_name: None,
            },
        );
    }

    fn put(&mut self, type_id: TypeId, slot: Slot) {
        let name = slot.type_name;
        if self.slots.insert(type_id, slot).is_some() && cfg!(debug_assertions) {
            ::tracing::warn!(
                type_name = name.unwrap_or("<shared value>"),
                "a run extension of this type was already set and has been replaced"
            );
        }
    }

    /// Borrow the value of type `T`, if there is one.
    pub fn get<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.slots
            .get(&TypeId::of::<T>())?
            .value
            .downcast_ref::<T>()
    }

    /// Whether a value of type `T` is present.
    pub fn contains<T: Any + Send + Sync>(&self) -> bool {
        self.slots.contains_key(&TypeId::of::<T>())
    }

    /// Number of stored values.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether nothing is stored.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Names of the stored types that are known, sorted, for error messages.
    pub(crate) fn known_type_names(&self) -> Vec<&'static str> {
        let mut names: Vec<_> = self.slots.values().filter_map(|s| s.type_name).collect();
        names.sort_unstable();
        names
    }
}

impl std::fmt::Debug for Extensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Extensions")
            .field("len", &self.len())
            .field("types", &self.known_type_names())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Tenant(&'static str);
    struct Counter(u32);

    #[test]
    fn values_are_looked_up_by_type() {
        let ext = Extensions::new().with(Tenant("acme")).with(Counter(3));
        assert_eq!(ext.get::<Tenant>(), Some(&Tenant("acme")));
        assert_eq!(ext.get::<Counter>().map(|c| c.0), Some(3));
        assert!(ext.get::<u8>().is_none());
        assert_eq!(ext.len(), 2);
    }

    #[test]
    fn a_second_value_of_the_same_type_replaces_the_first() {
        let ext = Extensions::new().with(Counter(1)).with(Counter(2));
        assert_eq!(ext.get::<Counter>().map(|c| c.0), Some(2));
        assert_eq!(ext.len(), 1);
    }

    #[test]
    fn a_shared_value_lands_in_the_slot_of_its_concrete_type() {
        let mut ext = Extensions::new();
        ext.insert_shared(Arc::new(Tenant("acme")));
        assert_eq!(ext.get::<Tenant>(), Some(&Tenant("acme")));
        assert!(!ext.contains::<Arc<Tenant>>());
    }

    #[test]
    fn clones_share_values() {
        let ext = Extensions::new().with(Tenant("acme"));
        let copy = ext.clone();
        assert!(std::ptr::eq(
            ext.get::<Tenant>().unwrap(),
            copy.get::<Tenant>().unwrap()
        ));
    }

    #[test]
    fn debug_lists_known_types() {
        let ext = Extensions::new().with(Tenant("acme"));
        let shown = format!("{ext:?}");
        assert!(shown.contains("Tenant"), "{shown}");
    }
}
