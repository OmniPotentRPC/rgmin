`Error` is `#[non_exhaustive]` and gains `AcceptStalled`.
That breaks code outside the crate that matches `Error` without a wildcard arm.
Add a `_` arm, and later variants need no change.
