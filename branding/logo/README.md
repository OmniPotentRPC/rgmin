# rgmin branding

Same LODE indigo / amber family as landfold, linkcell, and
readcon-core. The geometry is an hourglass.

**rgmin** is the local-minimization crate of the OmniPotentRPC
family (rgmin, rgsaddle, rgpot). Solvers live in Rust. C and C++
reach them through a narrow waist. The wordmark does not name
the objective crate.

## Concept

- **Upper chamber** — a potential well (nested contours).
- **Waist** — the C ABI (`rgmin_solver_t` / `rgmin_minimize`). Amber
  ring.
- **Lower chamber** — a unit sphere. Embedded manifolds retract
  here.
- **Amber path** — one descent through the waist onto the
  manifold.
- **Palette** — indigo / violet (`#1E1B4B`, `#312E81`, `#4F46E5`)
  + amber (`#FBBF24`, `#F59E0B`).

## Files

| File | Use |
| --- | --- |
| `rgmin-logo-light.svg` | Docs header (light), READMEs |
| `rgmin-logo-dark.svg` | Docs header (dark) |
| `rgmin-icon.svg` | Favicon / avatar (square) |
| `rgmin-notext-light.svg` | Navbar glyph (light) |
| `rgmin-notext-dark.svg` | Navbar glyph (dark) |
| `rgmin-logo.webp` | README raster (light wordmark) |
| `rgmin-logo-light.webp` | Sphinx `_static` |
| `rgmin-logo-dark.webp` | Sphinx `_static` |
| `rgmin-notext-light.webp` | Sphinx navbar |
| `rgmin-notext-dark.webp` | Sphinx navbar |
