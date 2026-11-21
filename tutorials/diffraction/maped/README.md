# MAPED

Align and merge a tilt series. Choose your data and backend; each notebook is
self-contained, with installation instructions and a preset backend.

| Data | CPU | NVIDIA CUDA | Apple MPS |
| --- | --- | --- | --- |
| Synthetic | [maped_synthetic_cpu.ipynb](maped_synthetic_cpu.ipynb) | [maped_synthetic_cuda.ipynb](maped_synthetic_cuda.ipynb) | [maped_synthetic_mps.ipynb](maped_synthetic_mps.ipynb) |
| Experimental | [maped_real_cpu.ipynb](maped_real_cpu.ipynb) | [maped_real_cuda.ipynb](maped_real_cuda.ipynb) | [maped_real_mps.ipynb](maped_real_mps.ipynb) |

These six notebooks show each alignment and merge stage. For one-call processing
of experimental data on MPS, use [maped_automatic.ipynb](maped_automatic.ipynb).
Seven notebooks total; choose one rather than running all seven.

## Install

Install **one widget package** for your GPU:

| Platform | Install | Automatically includes |
| --- | --- | --- |
| NVIDIA | `quantem.widget[cuda]` | `quantem.gpu[cuda]` |
| Apple Silicon | `quantem.widget[mps]` | `quantem.gpu[mps]` |

No separate GPU-package install is needed. Plain `quantem.widget` without an
extra does not install a GPU backend. CPU notebooks use native `quantem` only.

Follow [Conda or uv setup](setup/README.md) for the complete TestPyPI install
commands and Jupyter launch. Start with a synthetic notebook; no data download
is needed.

**Development prerequisite:** these MAPED APIs still require development builds;
released packages alone are not yet sufficient. The setup guide records the
current package constraints.

## Use MAPED

Start with loaded `tilts` having matching scan/detector shapes and devices.

```python
from quantem.diffraction import MAPED

maped = MAPED.from_datasets(tilts)
maped.align_diffraction()
maped.align_images(iterations=80)
maped.show_alignment(num_cols=4, size="medium", reference=0)
merged = maped.merge(crop=True)
maped.report()  # Table; use as_dict=True for a dictionary.
```

| Step | What it does |
| --- | --- |
| `align_diffraction()` | Matches mean diffraction patterns to measure shifts in **detector pixels**. |
| `align_images()` | Matches specimen images to measure shifts in **scan pixels**. |
| `merge()` | Applies both shifts and takes a weighted mean of the original diffraction data. |

Mean patterns/images guide alignment; merging uses the full 4D-STEM data.
Shifts use `(row, col)` coordinates.

**Automatic processing:** replace the three stage calls with
`merged = maped.run(iterations=80)`. It performs the same alignment and merge,
plus beam-center finding for display. You still acquire/load calibrated data
and inspect the report and merged patterns.

### Main settings

| Setting | Meaning |
| --- | --- |
| `iterations=80` | Image-alignment iterations; default **20**. Increase until the shifts stabilize. |
| `images=images` | Optional BF/ADF/DF or simultaneous HAADF images for `align_images()` or `run()`. Default: whole-detector mean images. |
| `precision=0.01` | Shift-search spacing in detector or scan pixels, respectively; not a guarantee of accuracy. |
| `crop=True` | Keep the scan area covered by every tilt. Defaults: `run()` → `True`, `merge()` → `False`. |

Supply one custom image per tilt, in tilt order, with matching scan shape,
sampling and orientation. For GPU ADF contrast, replace the image-alignment
call with:

```python
from quantem.gpu import detector

images = [detector.adf(tilt) for tilt in tilts]
maped.align_images(images=images, iterations=80)
```

## Inputs and results

Synthetic notebooks generate five 64 × 64 scans with 64 × 64 detector pixels,
known shifts, and Poisson counts. The illustrative scan step is 1 Å. This tests
registration, not microscope physics. `ground_truth.json` records the shifts
and a 0.2-pixel recovery target. No data download is needed.

Experimental notebooks use your Arina master, data, and metadata files. The CPU
notebook needs small, already-cropped and detector-corrected inputs, for example
64 × 64 scans; the native reader loads whole files. CUDA and MPS support the
full series with median replacement at stored bad detector pixels. Check your
calibration and detector corrections before interpreting the result.

Each notebook shows each plot and the report once. `show_tilts()` and
`show_alignment()` share `num_cols` and `size` options. The alignment view shows
Before, After and a signed difference from `reference`, the acquisition index
you choose. A difference also includes noise and tilt-dependent contrast; it is
not an alignment-error metric.

Experimental GPU notebooks save scaled uint16 HDF5. This is approximate: the
summary gives RMS rounding error, maximum absolute error in counts, and
clipping. CUDA and MPS reopen the saved codes as ANS-resident data, decoding
only requested regions. Encoded data plus temporary workspaces must fit in
device memory. Calibration and alignment metadata are saved with HDF5.
Use a new output filename for each run.

The experimental GPU examples use `release_tilts=True` during the merge.
Input GPU storage is released after its final use; original files are untouched.
Alignment plots use retained 2D images; subsequent pattern plots show the merged
result only. Reload the inputs if you want to merge them again.

Small CPU results remain float32 and save as native QuantEM ZIP. Synthetic GPU
examples use scaled uint16 ANS residency too. Scaled uint16 export currently supports HDF5, not
QEM. GPU notebooks include Show4DSTEM for inspecting the result.

Public notebooks have no experimental outputs, private paths, or widget state.

## Validation and source

The revised CUDA and MPS notebooks passed fresh-kernel execution with the
five-tilt synthetic fixture. Both real templates and the automatic MPS template
also passed with seven full-size 512 × 512 × 192 × 192 acquisitions. Input and
output ANS codecs and device selection were checked. Each overview, alignment
figure, pattern view and report appeared once. Saved outputs were visually
reviewed; widget construction passed, but these runs do not establish a fresh
interactive browser test. Both CPU templates also passed fresh-kernel execution
with the five-tilt synthetic fixture; full-size real CPU data was not tested.
These are source-build checks, not release-only installation validation.

Earlier versions of all seven notebooks were executed on development builds. Synthetic runs passed
on CPU, CUDA, and MPS: scan and detector shift errors were below 0.047 and
0.00051 pixels. Experimental checks used seven detector-corrected 64 × 64 crops
with 192 × 192 detector pixels. Automatic and step-by-step results matched
exactly within each backend, and save/reopen preserved values and MAPED metadata.
Both CPU notebooks ran with GPU/widget imports blocked.

Earlier full seven-tilt 512 × 512 × 192 × 192 experimental runs passed on CUDA
and MPS. Full-size real CPU processing was not run. These checks verify the
workflow, not experimental accuracy or real-time performance.

This workflow builds on Henry Bell’s [QuantEM PR #256](https://github.com/electronmicroscopy/quantem/pull/256).
Synthetic objects and shifts follow its [validation notebook](https://github.com/user-attachments/files/29561645/maped_torch_validation.ipynb).
