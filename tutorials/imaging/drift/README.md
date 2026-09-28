# Drift-correction tutorials

Executed tutorials for the public `quantem` drift-correction API. GitHub renders
the saved code and results directly, so each notebook can be reviewed before it
is downloaded and run.

## Start here

| Notebook | Dataset | Workflow |
| --- | --- | --- |
| [`2d_affine_co3o4.ipynb`](2d_affine_co3o4.ipynb) | Co₃O₄ HAADF | Automated 2D affine correction |
| [`2d_affine_silicon.ipynb`](2d_affine_silicon.ipynb) | Silicon HAADF | Automated 2D affine correction |
| [`2d_affine_srtio3.ipynb`](2d_affine_srtio3.ipynb) | SrTiO₃ HAADF | Automated 2D affine correction |
| [`2d_nonrigid_ws2.ipynb`](2d_nonrigid_ws2.ipynb) | WS₂ HAADF | Guided non-rigid correction |
| [`3d_xeds.ipynb`](3d_xeds.ipynb) | SrTiO₃ XEDS | Reference and strip correction of spectrum imaging |
| [`4d_4dstem.ipynb`](4d_4dstem.ipynb) | Gold 4D-STEM | Drift correction from virtual images |

Each notebook links to the exact public raw-data folder it needs. Download those
files into the notebook's `data/<dataset>/` directory, then run from top to
bottom. The notebooks install the reviewed `quantem` `drift-paper` branch and
the latest pre-release of `quantem.widget` from TestPyPI.

The notebooks default to `INTERACTIVE = True`. Set it to `False` to use the
same workflow with static figures. Device selection is automatic across CUDA,
Apple MPS, and CPU and is shown by `profile()` near the top of each notebook.

## Advanced tutorials

The [`advanced/`](advanced/) folder exposes diagnostic and explicit-parameter
workflows for users who need more control:

| Notebook | What it shows |
| --- | --- |
| [`2d_affine_explicit_co3o4.ipynb`](advanced/2d_affine_explicit_co3o4.ipynb) | Explicit full-resolution affine search on Co₃O₄ |
| [`2d_affine_explicit_silicon.ipynb`](advanced/2d_affine_explicit_silicon.ipynb) | Explicit full-resolution affine search on silicon |
| [`2d_affine_explicit_srtio3.ipynb`](advanced/2d_affine_explicit_srtio3.ipynb) | Explicit full-resolution affine search on SrTiO₃ |
| [`2d_affine_diagnostics_ws2_0070_0071.ipynb`](advanced/2d_affine_diagnostics_ws2_0070_0071.ipynb) | Regional affine diagnostics for a difficult WS₂ pair |
| [`2d_nonrigid_safety_ws2.ipynb`](advanced/2d_nonrigid_safety_ws2.ipynb) | Non-rigid bounds, diagnostics, and failure-mode checks |

All eleven notebooks were executed with the `live-env` kernel and were checked
for saved outputs and zero error cells before publication on this branch.

## Data credits

All datasets were acquired at the Stanford Nano Shared Facilities
(nano@stanford).

| Dataset | Credit |
| --- | --- |
| Silicon HAADF | Silicon sample provided by Samsung Electronics Corporation. |
| SrTiO₃ HAADF and XEDS | SrTiO₃ sample provided by Kevin Crust. |
| WS₂ HAADF | Sample provider to be confirmed with William Millsaps. |
| Co₃O₄ HAADF | Sample prepared by Jun Beom Hwang and Myoung Hwan Oh, KENTECH. |
| Gold 4D-STEM | Standard gold sample from the Stanford Nano Shared Facilities. |

We thank Pinaki Mukherjee for TEM instrument support and Berk Küçükoğlu and
Oliver Harder of DECTRIS for ARINA acquisition and automation support.
