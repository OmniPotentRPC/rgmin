//! Chainable `with_*` setters for the public configuration structs.
//!
//! Every configuration struct is `#[non_exhaustive]`: a caller outside
//! this crate starts from `Default::default()` or a named constructor
//! and chains setters, so a field added later is not a breaking change.
//! An `opt` field takes `impl Into<Option<T>>`, so `with_maxmove(0.2)`
//! sets `Some(0.2)` and `with_maxmove(None)` clears it.

macro_rules! with_setters {
    ($ty:ident $(<$lt:lifetime>)? { $($setter:ident => $field:ident : $kind:ident $fty:ty),* $(,)? }) => {
        impl$(<$lt>)? $ty$(<$lt>)? {
            $( with_setters!(@one $setter $field $kind $fty); )*
        }
    };
    (@one $setter:ident $field:ident val $fty:ty) => {
        #[doc = concat!("Replace [`Self::", stringify!($field), "`].")]
        ///
        /// Chain it after `default()` or a named constructor.
        #[must_use]
        pub fn $setter(mut self, $field: $fty) -> Self {
            self.$field = $field;
            self
        }
    };
    (@one $setter:ident $field:ident opt $fty:ty) => {
        #[doc = concat!("Replace [`Self::", stringify!($field), "`].")]
        ///
        /// A bare value is `Some`; `None` clears it.
        #[must_use]
        pub fn $setter(mut self, $field: impl Into<Option<$fty>>) -> Self {
            self.$field = $field.into();
            self
        }
    };
}
