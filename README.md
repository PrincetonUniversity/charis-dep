# CHARIS Pipeline & SPHERE / IFS Support

[![PyPI - Python Version](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-brightgreen)](https://github.com/PrincetonUniversity/charis-dep)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

> **Open-source, parallel, and publication-ready data-reduction for high-contrast integral-field spectrographs**

The **CHARIS pipeline** is the reference tool for transforming raw detector reads from the **Subaru/CHARIS** instrument into fully calibrated spectral data cubes—with uncertainties—ready for scientific analysis. The same core engine also powers the recommended **SPHERE/IFS** workflow via the companion project **[spherical](https://github.com/m-samland/spherical)**.

- **Peer-reviewed methods** • χ² cube extraction, lenslet-PSF forward modeling, correlated-read-noise removal ✨  
- **Fast** • Cython + OpenMP parallelism (< 3 min / cube on a laptop) ⚡  
- **Reproducible** • Strict version pinning & deterministic outputs 🔒

---

## Table of Contents
1. [Quick Start](#quick-start)  
2. [Installation](#installation)  
   * [OpenMP](#openmp)  
   * [Calibration data](#calibration-data)  
3. [Requirements & Dependencies](#requirements--dependencies)  
4. [Instrument Workflows](#instrument-workflows)  
   * [CHARIS](#charis-workflow)  
   * [SPHERE / IFS](#sphereifs-workflow)  
5. [System & Performance Notes](#system--performance-notes)  
6. [Citing](#citing)  
7. [Contributing](#contributing)  
8. [License](#license)

Release notes live in [`CHANGELOG.md`](./CHANGELOG.md).

---

## Quick Start
```bash
# Create a fresh virtual environment (recommended)
python -m venv charisenv && source charisenv/bin/activate

# Install the latest stable pipeline
pip install git+https://github.com/PrincetonUniversity/charis-dep@main

# Build a calibration (CHARIS example; interactive)
buildcal /path/to/CRSA00000001.fits
# Optional: pass darks and mode info
buildcal /path/to/CRSA00000001.fits 1320 J /path/to/darks/*fits

# Extract a data cube
extractcube /path/to/CRSA00000000.fits my_config.ini
```
For SPHERE/IFS data, jump directly to the [SPHERE workflow](#sphereifs-workflow).

---

## Installation
1. **Python ≥ 3.10** is required. We test against 3.10 – 3.13.  
2. Set up an isolated environment:  
   * **venv**
     ```bash
     python3.12 -m venv charisenv
     source charisenv/bin/activate
     ```
   * **Conda**
     ```bash
     conda create -n charisenv python=3.12
     conda activate charisenv
     ```
3. Install the pipeline:
   ```bash
   pip install git+https://github.com/PrincetonUniversity/charis-dep@main
   ```
4. (Developers) Editable install:
   ```bash
   git clone https://github.com/PrincetonUniversity/charis-dep.git
   cd charis-dep && pip install -e .
   ```
   Or with [pixi](https://pixi.sh), which is what the project is developed against:
   ```bash
   pixi install -e dev      # test + notebook tooling
   pixi run -e test test    # pytest
   pixi run -e test lint    # ruff check src/charis/
   ```

### OpenMP
The compiled extensions run their inner loops in parallel with OpenMP (the `maxproc` setting).
What the build needs depends on the platform.

* **Linux**: nothing extra with gcc, which ships OpenMP.
* **macOS**: Apple's clang has no OpenMP runtime of its own.
  Either run `brew install libomp` before installing,
  or install into a conda or pixi environment that provides `libomp` (conda-forge `llvm-openmp`).
  The build looks for `omp.h` and `libomp.dylib` in the active environment first,
  then in Homebrew (`/opt/homebrew/opt/libomp` on Apple Silicon, `/usr/local/opt/libomp` on Intel).

If OpenMP is not found, or the OpenMP build fails, charis is built serially and prints a warning.
`pip` only shows that warning with `pip install -v`, so check the result after installing.

```bash
python -c "import charis; print(charis.openmp_enabled())"   # True = parallel build
```

Two environment variables control the build.

| Variable | Effect |
|----------|--------|
| `CHARIS_REQUIRE_OPENMP=1` | Fail the build instead of falling back to a serial build. |
| `CHARIS_NO_OPENMP=1` | Build serially on purpose, without looking for OpenMP. |

For example, run `CHARIS_REQUIRE_OPENMP=1 pip install -v .` in a clone.

### Calibration data
The static calibration files (bad-pixel masks, flats, PSFlet templates, wavelength solutions,
background templates) are not part of the package. charis downloads them per instrument from
Zenodo the first time an instrument is used (SPHERE about 86 MB, CHARIS about 71 MB), checks the
md5, and keeps them in a cache. Later runs need no network.

| Instrument | Data version | DOI |
|------------|--------------|-----|
| SPHERE IFS | 2026.10.0 | [10.5281/zenodo.23640607](https://doi.org/10.5281/zenodo.23640607) |
| CHARIS | 2026.10.0 | [10.5281/zenodo.23640685](https://doi.org/10.5281/zenodo.23640685) |

The cache is `$CHARIS_CALIBRATION_DIR` if set, otherwise `~/.cache/charis/calibrations`
(`$XDG_CACHE_HOME/charis/calibrations`), with one folder per data version. Each cube records the
data version and DOI in its header (`CALVERS`, `CALDOI`).

On a cluster whose compute nodes have no internet access, download once on a login node into a
shared directory and point every job at it:

```bash
export CHARIS_CALIBRATION_DIR=/shared/path/charis-calibrations
charis-calibrations fetch SPHERE      # or CHARIS, or all
charis-calibrations path              # prints the directory in use
```

The directory can be read-only for the jobs. Each charis release pins the exact calibration
version it was tested with, so new calibrations only arrive with a charis update. Old version
folders are never deleted automatically; remove them by hand when no installed charis uses them.
To use your own calibration files instead, pass `static_calibdir` to the instrument.

---

## Requirements & Dependencies
| Kind | Packages |
|------|----------|
| **Core** | `numpy` · `scipy` · `astropy` · `pandas` |
| **Visualization** | `matplotlib` · `bokeh` |
| **Acceleration** | `cython` · OpenMP-capable C compiler |
| **Utilities** | `tqdm` · `bottleneck` · `psutil` |

> **Memory** ≥ 2 GB per extraction; ≥ 2 GB / core (≥ 4 GB total) when rebuilding calibrations.

---

## Instrument Workflows

### CHARIS Workflow
| Step | Command | Notes |
|------|---------|-------|
| **1. Build calibration** | `buildcal <monochromatic_flat.fits> <λ[nm]> <mode>` | Accepts optional dark/background frames. If wavelength & mode are encoded in the header, omit them. |
| **2. Configure extraction** | Copy & edit [`src/charis/sample.ini`](./src/charis/sample.ini) | Tune bad-pixel masks, cube size, etc. |
| **3. Extract cube** | `extractcube <raw_reads.fits> <config.ini>` | Generates a 4‑HDU FITS: header · cube · inverse-variance · raw-header. |

### SPHERE/IFS Workflow
For SPHERE/IFS we recommend the **[spherical](https://github.com/m-samland/spherical)** wrapper, which automates the entire process:

```bash
pip install git+https://github.com/m-samland/spherical.git
```
`spherical` handles data discovery, download, calibration, cube extraction, and post-processing—using this pipeline under the hood for the heavy lifting. Please refer to the examples in the repository for details.

---

## System & Performance Notes
### macOS (Apple Silicon & Intel)
```bash
brew install libomp          # enable multi-core acceleration
pip install git+https://github.com/PrincetonUniversity/charis-dep@main
```
Without OpenMP the installer falls back to a serial build, which is much slower on large datasets.
See [OpenMP](#openmp) for how to check the build and how to make a missing OpenMP an error.

---

## Citing
If this pipeline contributes to your research, please cite both foundational papers:

- **[Brandt et al. 2017](https://ui.adsabs.harvard.edu/abs/2017JATIS...3d8002B/abstract)**, *JATIS* 3, 4, 8002  
  DOI: 10.1117/1.JATIS.3.4.048002
- **[Samland et al. 2022](https://ui.adsabs.harvard.edu/abs/2022A%26A...668A..84S/abstract)**, *A&A* 668, A84

If you publish results based on the shipped calibration files, please also cite the data
version you used ([SPHERE](https://doi.org/10.5281/zenodo.23640607),
[CHARIS](https://doi.org/10.5281/zenodo.23640685); the cube header's `CALDOI` names it).

The ADS entries are linked in the bibliography below.

---

## Contributing
We welcome issues & PRs—large or small.

1. **Fork** → **create a feature branch** → **commit** → **open a PR** against `devel`.  
2. Run `pixi run -e test test` and `pixi run -e test lint` before opening the PR.  
3. Note user-visible changes in [`CHANGELOG.md`](./CHANGELOG.md) under `[Unreleased]`.  
4. New here? Open an issue or email [Tim Brandt](mailto:timothy.d.brandt@gmail.com) or [Matthias Samland](mailto:matthias.samland@gmail.com) for guidance.

---

## License
This project is distributed under the **BSD-3-Clause License**—see [`LICENSE`](LICENSE) for the full text.

---

### Bibliography
* [Brandt, T. D., *et al.* 2017](https://ui.adsabs.harvard.edu/abs/2017JATIS...3d8002B/abstract), "CHARIS Data Reduction Pipeline", **JATIS**, 3, 048002  
* [Samland, M., *et al.* 2022](https://ui.adsabs.harvard.edu/abs/2022A%26A...668A..84S/abstract), "A New SPHERE IFS Pipeline Based on CHARIS", **A&A**, 668, A84
