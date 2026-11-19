# Report source

LaTeX source of the method description submitted to OpenReview. It follows the
section skeleton of the organisers' report template (linked from the OpenReview
submission form) and the official NeurIPS 2026 style.

To build it, put `neurips_2026.sty` from the NeurIPS 2026 formatting
instructions (https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip)
next to `method.tex` and run `tectonic method.tex` or `latexmk -pdf method.tex`
(BibTeX reads `refs.bib`). The author line is read from an optional
`_author.tex`; without it the report is credited to team `junhyun-dev`.
`probe-result.tex` holds the values read back from the probe submission.
