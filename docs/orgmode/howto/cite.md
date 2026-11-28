

# Cite rgmin

This page gives a BibTeX entry for rgmin that names the version you ran.


## Get the entry

`CITATION.cff` at the top of the repository is the citation record, in the Citation File Format (CFF).
It names version 0.8.0, dated 22 November 2026.
cffconvert turns it into BibTeX.
Run it at the top of a checkout:

```bash
pip install cffconvert
cffconvert -f bibtex
```

cffconvert 2.0.0 prints:

    @misc{YourReferenceHere,
    author = {Goswami, Rohit},
    doi = {10.5281/zenodo.24437655},
    month = {11},
    title = {rgmin},
    url = {https://github.com/OmniPotentRPC/rgmin},
    year = {2026}
    }

The month and year come from the release date, and the entry has no version.
Its DOI is the Zenodo concept DOI, <https://doi.org/10.5281/zenodo.24437655>, which resolves to the latest release.
The record for 0.8.0 is <https://doi.org/10.5281/zenodo.24469714>.
`codemeta.json` (CodeMeta 2.0) and `.zenodo.json` carry the same metadata for software indexes and for the Zenodo archive.
The "Cite this repository" button on GitHub reads the record on the default branch, `main`.


## Find the version you ran

In the project that depends on rgmin, run:

```bash
cargo tree -p rgmin --depth 0
```

With the release from crates.io, it prints:

    rgmin v0.8.0

With the `develop` branch, the short commit follows the `#`:

    rgmin v0.6.0 (https://github.com/OmniPotentRPC/rgmin?branch=develop#113fb592)

The version in `Cargo.toml` on `develop` does not change with each commit, so on that route cite the commit.
`Cargo.lock` has the full commit:

```bash
grep -A2 '^name = "rgmin"' Cargo.lock
```

    name = "rgmin"
    version = "0.6.0"
    source = "git+https://github.com/OmniPotentRPC/rgmin?branch=develop#113fb592707406a6d621e4da6d8eea243abce968"

In the rgmin checkout that `Cargo.lock` names, `git rev-parse HEAD` prints the same full commit.


## Put the version in the entry

Give the entry a key, and add the version.
The `plain` style of BibTeX prints a `note` field, but it drops `version` and `url`, so put the version in `note`.
Braces around `rgmin` in the title stop a style from capitalizing the name.
For the 0.8.0 release:

```bibtex
@misc{rgmin,
author = {Goswami, Rohit},
title = {{rgmin}},
url = {https://github.com/OmniPotentRPC/rgmin},
doi = {10.5281/zenodo.24469714},
note = {Version 0.8.0},
year = {2026}
}
```

On the `develop` route, the note names the full commit, such as `note = {Commit 113fb592707406a6d621e4da6d8eea243abce968 of the develop branch}`.


## Cite the published work too

The methods in rgmin come from published work.
The explanation pages name the book or paper that each derivation follows, and lists the works that the docs cite.
Cite the sources of the methods you ran, along with rgmin.
