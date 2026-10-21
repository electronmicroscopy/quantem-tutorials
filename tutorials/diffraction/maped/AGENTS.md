# MAPED tutorial boundaries

- Keep the CPU workflow runnable with QuantEM alone; GPU packages are not
  requirements for CPU examples. Use `report(as_dict=True)` without pandas.
- GPU examples may load through `quantem.gpu`; keep ANS and storage choices
  in that loading layer, not in MAPED alignment, merging, or plot code.
- Show each figure and report once. Keep layout options consistent across
  `show_tilts` and `show_alignment`; keep numerical diagnostics out of cells
  when the library already provides them.
- State measured scaled-uint16 rounding errors without claiming losslessness.
  Keep public notebooks free of private paths, data, outputs, and widget state.
