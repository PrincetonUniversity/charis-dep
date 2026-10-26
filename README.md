# CHARIS Pipeline & SPHERE / IFS Support

[![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-brightgreen)](https://pypi.org/p/charis)
[![CI](https://github.com/PrincetonUniversity/charis-dep/actions/workflows/ci.yml/badge.svg)](https://github.com/PrincetonUniversity/charis-dep/actions/workflows/ci.yml)
[![License: BSD-2-Clause-Views](https://img.shields.io/badge/license-BSD--2--Clause--Views-blue.svg)](LICENSE)

> **Open-source, parallel, and publication-ready data-reduction for high-contrast integral-field spectrographs**

The **CHARIS pipeline** turns raw detector reads from the **Subaru/CHARIS** instrument into fully calibrated spectral data cubes with inverse-variance maps, ready for scientific analysis. The same engine reduces **VLT/SPHERE IFS** data, for which we recommend the companion project **[spherical](https://github.com/m-samland/spherical)**.

- **Peer-reviewed methods** • χ² cube extraction, lenslet-PSF forward modeling, correlated-read-noise removal ✨  
- **Fast** • Cython + OpenMP parallelism (< 3 min / cube on a laptop) ⚡  
- **Traceable calibrations** • versioned calibration data with a DOI, recorded in every cube header 🔒

---

## Table of Contents
1. [Quick Start](#quick-start)  
2. [Installation](#installation)  
   * [OpenMP](#openmp)  
   * [Calibration data](#calibration-data)  
3. [Instrument Workflows](#instrument-workflows)  
   * [CHARIS](#charis-workflow)  
   * [SPHERE / IFS](#sphereifs-workflow)  
4. [Command-line Tools](#command-line-tools)  
5. [Requirements](#requirements)  
6. [Citing](#citing)  
7. [Contributing](#contributing)  
8. [License](#license)

Release notes live in [`CHANGELOG.md`](./CHANGELOG.md).

---

## Quick Start
```bash
# Create a fresh virtual environment (recommended)
python -m venv charisenv && source charisenv/bin/activate

# Install the pipeline
pip install charis

# Build a calibration from a CHARIS narrow-band flat (interactive), optionally with darks
buildcal /path/to/CRSA00000001.fits /path/to/darks/CRSA*.fits

# Extract data cubes, configured by a copy of src/charis/sample.ini
extractcube /path/to/CRSA0000*.fits my_config.ini
```
For SPHERE/IFS data, jump directly to the [SPHERE workflow](#sphereifs-workflow).

---

## Installation
1. **Python 3.11 to 3.13** is required.  
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
   pip install charis
   ```
   This installs a prebuilt wheel with OpenMP on Linux (x86_64, aarch64) and on macOS 14 or newer (Apple Silicon, Intel).
   Elsewhere pip builds charis from source, which needs a C compiler and OpenMP (see below).
   For the development version, install from GitHub instead:
   `pip install git+https://github.com/PrincetonUniversity/charis-dep@devel`.
4. (Developers) Editable install:
   ```bash
   git clone https://github.com/PrincetonUniversity/charis-dep.git
   cd charis-dep && pip install -e ".[test]"
   ```
   Or with [pixi](https://pixi.sh), which is what the project is developed against:
   ```bash
   pixi install -e dev      # test + notebook tooling
   pixi run -e test test    # pytest
   pixi run -e test lint    # ruff check src/charis/
   ```

### OpenMP
The compiled extensions run their inner loops in parallel with OpenMP; the `maxcpus` setting of the
configuration file sets the number of threads.
The wheels on PyPI include OpenMP. When building from source, what the build needs depends on the platform.

* **Linux**: nothing extra with gcc, which ships OpenMP.
* **macOS**: Apple's clang has no OpenMP runtime of its own.
  Either run `brew install libomp` before installing,
  or install into a conda or pixi environment that provides `libomp` (conda-forge `llvm-openmp`).
  The build looks for `omp.h` and `libomp.dylib` in the active environment first,
  then in Homebrew (`/opt/homebrew/opt/libomp` on Apple Silicon, `/usr/local/opt/libomp` on Intel).

If OpenMP is not found, or the OpenMP build fails, charis is built serially and prints a warning.
A serial build works but is much slower on large datasets.
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

## Instrument Workflows

### CHARIS Workflow
| Step | Command | Notes |
|------|---------|-------|
| **1. Build calibration** | `buildcal <narrow-band_flat.fits> [<dark.fits> ...]` | Reads the band and wavelength from the header and asks for them if they are missing. Asks whether to oversample the PSFlet templates (needed for `fitshift`) and how many threads to use. Writes the calibration files into the current directory. |
| **2. Configure extraction** | Copy & edit [`src/charis/sample.ini`](./src/charis/sample.ini) | Set `calibdir` to the directory from step 1; every option is documented in the file. |
| **3. Extract cubes** | `extractcube <raw_reads*.fits> <config.ini>` | Writes one `*_cube.fits` per input file: an empty primary HDU with the cube header, then the cube, its inverse variance and the raw file's header. |

### SPHERE/IFS Workflow
For SPHERE/IFS we recommend the **[spherical](https://github.com/m-samland/spherical)** pipeline, which installs charis and automates the entire process:

```bash
pip install "spherical[pipeline] @ git+https://github.com/m-samland/spherical.git"
```
spherical handles data discovery, download, calibration, cube extraction, and post-processing, using this pipeline for the cube extraction.
See its README and example scripts for details.

---

## Command-line Tools
| Command | Purpose |
|---------|---------|
| `buildcal` | Build a CHARIS calibration set from a narrow-band flat (see [CHARIS Workflow](#charis-workflow)). |
| `extractcube` | Extract cubes from raw files with a configuration file. |
| `charis-calibrations` | `fetch` the [calibration data](#calibration-data) of an instrument ahead of time, or print its `path`. |
| `hexplot` | Interactive Bokeh viewer of a SPHERE cube on its native hexagonal lenslet grid. |

---

## Requirements
pip installs the dependencies (`numpy`, `scipy`, `astropy`, `pandas`, `matplotlib`, `bokeh`,
`bottleneck`, `tqdm`, `psutil`, `future`) automatically.
Building from source also needs Cython and an OpenMP-capable C compiler (see [OpenMP](#openmp)).

> **Memory** ≥ 2 GB per extraction; ≥ 2 GB / core (≥ 4 GB total) when rebuilding calibrations.

---

## Citing
If this pipeline contributes to your research, please cite the paper for your instrument:

- **CHARIS:** [Brandt et al. 2017](https://ui.adsabs.harvard.edu/abs/2017JATIS...3d8002B/abstract), "Data reduction pipeline for the CHARIS integral-field spectrograph I: detector readout calibration and data cube extraction", *JATIS* 3, 048002, DOI: [10.1117/1.JATIS.3.4.048002](https://doi.org/10.1117/1.JATIS.3.4.048002)
- **SPHERE IFS:** [Samland et al. 2022](https://ui.adsabs.harvard.edu/abs/2022A%26A...668A..84S/abstract), "Spectral cube extraction for the VLT/SPHERE IFS: Open-source pipeline with full forward modeling and improved sensitivity", *A&A* 668, A84

If you publish results based on the shipped calibration files, please also cite the data
version you used ([SPHERE](https://doi.org/10.5281/zenodo.23640607),
[CHARIS](https://doi.org/10.5281/zenodo.23640685); the cube header's `CALDOI` names it).

---

## Contributing
We welcome issues & PRs, large or small.

1. **Fork** → **create a feature branch** → **commit** → **open a PR** against `devel`.  
2. Run `pixi run -e test test` and `pixi run -e test lint` before opening the PR.  
3. Note user-visible changes in [`CHANGELOG.md`](./CHANGELOG.md) under `[Unreleased]`.  
4. New here? Open an issue or email [Tim Brandt](mailto:timothy.d.brandt@gmail.com) or [Matthias Samland](mailto:matthias.samland@gmail.com) for guidance.

Maintainers: the release steps are in [`RELEASING.md`](./RELEASING.md).

---

## License
This project is distributed under the **BSD-2-Clause-Views License** (the two-clause BSD license with the FreeBSD views paragraph); see [`LICENSE`](LICENSE) for the full text.
