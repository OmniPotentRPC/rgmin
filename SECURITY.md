# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| Latest `main` / newest `vX.Y.Z` tag | Yes (fixes land on `main` first) |
| Older minor lines | Best-effort only; please upgrade to the latest tag |

rgmin is published on crates.io and tagged `vX.Y.Z` on `main`.
Security fixes are released via normal semver bumps and a `v*` tag.

## Reporting a vulnerability

Please do not open a public GitHub issue for unfixed vulnerabilities.

1. Email the maintainer, Rohit Goswami, at rgoswami@ieee.org, or
2. Use GitHub Security Advisories / private vulnerability reporting if
   enabled for this repository.

Include: affected versions/tags, reproduction or impact, and whether a
fix is already proposed.

## Scope

In scope: memory-safety issues reachable through the safe Rust API or
the documented C ABI contract, unsafe defaults in public APIs, and
soundness holes in the `unsafe` blocks behind the DLPack waist.

Out of scope (unless trivially fixed): misuse of the C ABI outside its
documented contract (e.g. freeing borrowed tensors, violating the
callback lifetimes the headers state), and issues only in unreleased
experimental branches.
