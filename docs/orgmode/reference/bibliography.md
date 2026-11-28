

# Bibliography

Validated DOIs only. The Sphinx bibliography is
`docs/source/references.bib`, consumed by `sphinxcontrib-bibtex`
(biblatex .bib, unsrt). New citations enter through ookcite
(`validate_doi`, then `add_to_collection` on `rgmin-docs`), never
as invented keys or a markdown table. Regenerate the .bib with
`export_collection`. Narrative pages cite with the RST role
`` :cite:`key` ``.

Packwood, Kermode, Mones, Bernstein, Woolley, Gould, Ortner, and
Csányi, *A universal preconditioner for simulating condensed phase
materials*, J. Chem. Phys. 144, 164109 (2016),
[10.1063/1.4947024](https://doi.org/10.1063/1.4947024).
