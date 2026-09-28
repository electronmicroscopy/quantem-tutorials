# Advanced drift-correction tutorials

These executed notebooks complement the automated workflows in the
[`drift/`](../) folder. Use them when you need explicit affine-search settings,
regional diagnostics, or non-rigid safety checks.

| Notebook | What it shows |
| --- | --- |
| [`2d_affine_explicit_co3o4.ipynb`](2d_affine_explicit_co3o4.ipynb) | Full-resolution explicit affine search on Co₃O₄ |
| [`2d_affine_explicit_silicon.ipynb`](2d_affine_explicit_silicon.ipynb) | Full-resolution explicit affine search on silicon |
| [`2d_affine_explicit_srtio3.ipynb`](2d_affine_explicit_srtio3.ipynb) | Full-resolution explicit affine search on SrTiO₃ |
| [`2d_affine_diagnostics_ws2_0070_0071.ipynb`](2d_affine_diagnostics_ws2_0070_0071.ipynb) | Regional affine ambiguity and trusted-region diagnostics |
| [`2d_nonrigid_safety_ws2.ipynb`](2d_nonrigid_safety_ws2.ipynb) | Motion-bound diagnosis and unsupported-bending checks |

The notebooks use the same public data folders and `live-env` setup documented
in the parent README. They default to interactive `quantem.widget` views; set
`INTERACTIVE = False` for static output.
