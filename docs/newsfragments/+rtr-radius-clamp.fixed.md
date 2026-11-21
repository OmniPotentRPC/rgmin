`RtrRadius::update` and `rgmin_rtr_radius_update` now clamp a radius above `radius_max` to the maximum before the update, so a successful call never leaves the radius above it.
