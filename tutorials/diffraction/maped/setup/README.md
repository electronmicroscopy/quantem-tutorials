# MAPED setup

Choose **Conda** or **uv**, then choose your backend. These instructions use
Python 3.12; CUDA needs an NVIDIA GPU with a compatible driver, and MPS needs
an Apple Silicon Mac. CPU notebooks do not need GPU packages.

**Release status:** the MAPED workflow still requires development builds of
QuantEM and, for GPU use, `quantem.gpu`. The package installs below set up the
notebook environment but do not yet supply those APIs. Published widget rc41
extras currently select GPU rc15. The MAPED notebooks were validated in
development environments, not fresh release-only installs.

## Get the notebooks

On the repository page, choose **Code → Download ZIP**, extract it, and open a
terminal in `tutorials/diffraction/maped`. Keep the `setup` folder beside the
notebooks. Start with the synthetic notebook for your backend; it creates its
own small dataset.

## Conda (recommended)

Install [Miniforge](https://github.com/conda-forge/miniforge#install), reopen your
terminal, and run the **create** and **activate** commands for one row below.
The environment file installs Python, JupyterLab, and the selected packages.
You do not need a separate pip install.

| Backend | Create once | Activate each session |
| --- | --- | --- |
| CPU | `conda env create -f setup/cpu.yml` | `conda activate maped-cpu` |
| NVIDIA CUDA | `conda env create -f setup/cuda.yml` | `conda activate maped-cuda` |
| Apple MPS | `conda env create -f setup/mps.yml` | `conda activate maped-mps` |

The GPU environments install `quantem.widget[cuda]` or `quantem.widget[mps]`,
which brings in the corresponding `quantem.gpu` backend automatically.

To update an existing environment, use its same file:

```bash
# Choose one:
conda env update -f setup/cpu.yml --prune
conda env update -f setup/cuda.yml --prune
conda env update -f setup/mps.yml --prune
```

## uv (alternative)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create
an isolated environment. `--seed` includes pip so the install commands also
work inside Jupyter.

```bash
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
Run **one** install command for your backend:

```bash
# CPU
python -m pip install --upgrade quantem pandas jupyterlab

# NVIDIA CUDA
python -m pip install --upgrade --pre -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "quantem.widget[cuda]" quantem pandas jupyterlab

# Apple MPS
python -m pip install --upgrade --pre -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "quantem.widget[mps]" quantem pandas jupyterlab
```

## Open and run

With your chosen environment active, launch Jupyter from the MAPED directory:

```bash
python -m jupyterlab
```

1. Open `maped_synthetic_cpu.ipynb`, `maped_synthetic_cuda.ipynb`, or
   `maped_synthetic_mps.ipynb` for your backend. Choose the **Python 3** kernel.
2. Select **Run → Run All Cells**. Skip the commented package-install cell when
   using an environment already prepared above. GPU notebooks show the device
   through `qw.profile()`; verify it before processing.
3. Inspect the merged diffraction patterns and shift table. Once the synthetic
   example works, open the corresponding `maped_real_*` notebook and set its
   `folder` to your tilt series. Use small inputs for CPU.

Save the notebook with **Ctrl+S** (or **Cmd+S**). To stop Jupyter, press **Ctrl+C**
in its terminal and confirm shutdown. Next time, activate the same environment
and run `python -m jupyterlab` again; you do not need to reinstall.
