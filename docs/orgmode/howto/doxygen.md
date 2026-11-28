

# Doxygen with doxyYoda

The C and C++ hourglass is documented by Doxygen and themed with
[doxyYoda](https://github.com/HaoZeke/doxyYoda) 0.2.2.


## Fetch the theme

```bash
scripts/get_doxyyoda.sh
```

This extracts the release tarball into `docs/doxyYoda/` (gitignored).


## Build the HTML

```bash
cd docs/source
doxygen Doxyfile_doxyyoda.cfg
```

Output lands in `docs/doxygen-html/html/`. The Doxyfile points at

-   `docs/doxyYoda/html/header.html`
-   `docs/doxyYoda/html/footer.html`
-   `docs/doxyYoda/css/doxyYoda.min.css`
-   `docs/doxyYoda/xml/doxyYoda.xml`


## What is extracted

`INPUT` is the public hourglass: `include/xts.h`, `include/xts/*.hpp`,
and `docs/source/mainpage.dox`. Rust rustdoc stays on `cargo doc`.
The orgmode book in `docs/orgmode/` is the narrative layer.
