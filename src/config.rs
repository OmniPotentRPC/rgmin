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
        #[doc = concat!("Replace [`", stringify!($field), "`](field@Self::", stringify!($field), ").")]
        ///
        /// Chain it after `default()` or a named constructor.
        #[must_use]
        pub fn $setter(mut self, $field: $fty) -> Self {
            self.$field = $field;
            self
        }
    };
    (@one $setter:ident $field:ident opt $fty:ty) => {
        #[doc = concat!("Replace [`", stringify!($field), "`](field@Self::", stringify!($field), ").")]
        ///
        /// A bare value is `Some`; `None` clears it.
        #[must_use]
        pub fn $setter(mut self, $field: impl Into<Option<$fty>>) -> Self {
            self.$field = $field.into();
            self
        }
    };
}

/// Two doctests per configuration type: the setter chain compiles
/// outside the crate and the struct literal does not. Stable rustdoc
/// passes a `compile_fail` block on any error, so the pair shares its
/// import, base value, field and value, and differs only in the literal.
/// `tests/config_setters.rs` fails when a type with setters has no entry.
#[cfg(doctest)]
mod literal_outside_crate {
    macro_rules! literal_fails {
        ($ty:ident, $import:literal, $base:literal, $field:ident = $value:literal) => {
            #[doc = concat!(
                            "```\nuse ", $import, ";\n",
                            "let v = ", $base, ".with_", stringify!($field), "(", $value, ");\n",
                            "let _ = &v.", stringify!($field), ";\n```"
                        )]
            #[doc = ""]
            #[doc = concat!(
                            "```compile_fail,E0639\nuse ", $import, ";\n",
                            "let v = ", stringify!($ty), " { ", stringify!($field), ": ", $value,
                            ", ..", $base, " };\n```"
                        )]
            struct $ty;
        };
    }

    literal_fails!(
        Control,
        "rgmin::Control",
        "Control::default()",
        maxiter = "7"
    );
    literal_fails!(
        NoiseFloor,
        "rgmin::NoiseFloor",
        "NoiseFloor::new(1e-3)",
        c = "2.0"
    );
    literal_fails!(
        ScgParams,
        "rgmin::ScgParams",
        "ScgParams::default()",
        sigma0 = "1e-5"
    );
    literal_fails!(
        ScgOptions,
        "rgmin::{ScgOptions, ScgStepTolerance}",
        "ScgOptions::default()",
        step_tolerance = "ScgStepTolerance::RelativeEuclidean"
    );
    literal_fails!(EonCg, "rgmin::EonCg", "EonCg::default()", max_iter = "5");
    literal_fails!(
        FireConfig,
        "rgmin::FireConfig",
        "FireConfig::ase()",
        dt = "0.2"
    );
    literal_fails!(
        Fire2Extras,
        "rgmin::Fire2Extras",
        "Fire2Extras::new(0.1)",
        n_delay = "5"
    );
    literal_fails!(
        QuickMinOptions,
        "rgmin::QuickMinOptions",
        "QuickMinOptions::default()",
        per_atom = "true"
    );
    #[cfg(feature = "highs")]
    literal_fails!(
        HighsStep,
        "rgmin::HighsStep",
        "HighsStep::default()",
        equalities = "Vec::new()"
    );
    literal_fails!(
        HighsOptions,
        "rgmin::{HighsOptions, HighsSolverKind}",
        "HighsOptions::default()",
        solver = "HighsSolverKind::Ipm"
    );
    literal_fails!(
        TrustRegion,
        "rgmin::TrustRegion",
        "TrustRegion::new(0.1)",
        maxiter = "7"
    );
    literal_fails!(
        RtrRadius,
        "rgmin::rtr::RtrRadius",
        "RtrRadius::new(1.0)",
        radius = "0.5"
    );
    literal_fails!(
        IrcTrust,
        "rgmin::IrcTrust",
        "IrcTrust::new(ndarray::Array1::zeros(3), ndarray::Array1::ones(3), 0.1)",
        dx = "0.2"
    );
    literal_fails!(
        EigenParams,
        "rgmin::EigenParams",
        "EigenParams::default()",
        nev = "2"
    );
    literal_fails!(
        ElpaParams,
        "rgmin::ElpaParams",
        "ElpaParams::default()",
        nblk = "16"
    );
    literal_fails!(
        EigenExaParams,
        "rgmin::{EigenExaAlgo, EigenExaParams}",
        "EigenExaParams::default()",
        algo = "EigenExaAlgo::Sx"
    );
    literal_fails!(
        SlepcParams,
        "rgmin::{SlepcParams, SlepcStKind}",
        "SlepcParams::default()",
        st = "SlepcStKind::Shift"
    );
    literal_fails!(
        DlaFutureParams,
        "rgmin::DlaFutureParams",
        "DlaFutureParams::default()",
        begin = "1"
    );
    literal_fails!(
        ExpParams,
        "rgmin::ExpParams",
        "ExpParams::default()",
        a = "2.0"
    );
    literal_fails!(
        Neighbours,
        "rgmin::Neighbours",
        "Neighbours::default()",
        i = "vec![0]"
    );
    literal_fails!(
        LindhParams,
        "rgmin::LindhParams",
        "LindhParams::default()",
        c_stab = "0.2"
    );
    literal_fails!(
        FfParams,
        "rgmin::FfParams",
        "FfParams::default()",
        c_stab = "0.2"
    );
    literal_fails!(
        FfTerms,
        "rgmin::FfTerms",
        "FfTerms::default()",
        bonds = "&[]"
    );
    literal_fails!(
        LindhSpec,
        "rgmin::LindhSpec",
        "LindhSpec::new(vec![1], 0.1, [10.0; 3])",
        c_stab = "0.2"
    );
    literal_fails!(Lbfgs, "rgmin::Lbfgs", "Lbfgs::default()", gtol = "0.5");
    literal_fails!(
        RigidRequest,
        "rgmin::{RigidMode, RigidRequest}",
        "RigidRequest::cartesian(&[0.0; 3], RigidMode::Auto)",
        mass_metric = "true"
    );
    literal_fails!(
        Report,
        "rgmin::Report",
        "Report::new(0.0, ndarray::Array1::zeros(1))",
        steps = "3"
    );
}
