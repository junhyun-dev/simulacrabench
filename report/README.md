# Report source

LaTeX source of the 4-page method description submitted to OpenReview.

To build it, put the organisers' `simulacrabench-template.sty` (linked from the
OpenReview submission form) next to `method.tex` and run any LaTeX engine, for
example `tectonic method.tex` or `latexmk -pdf method.tex`. Without the template
the file falls back to NeurIPS-sized pages. The author line is read from an
optional `_author.tex`; without it the report is credited to team `junhyun-dev`.
`probe-result.tex` holds the values read back from the probe submission.
