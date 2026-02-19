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
3. [Requirements & Dependencies](#requirements--dependencies)  
4. [Instrument Workflows](#instrument-workflows)  
   * [CHARIS](#charis-workflow)  
   * [SPHERE / IFS](#sphereifs-workflow)  
5. [System & Performance Notes](#system--performance-notes)  
6. [Citing](#citing)  
7. [Contributing](#contributing)  
8. [License](#license)

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
| **2. Configure extraction** | Copy & edit [`sample.ini`](./sample.ini) | Tune bad-pixel masks, cube size, etc. |
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
The installer automatically falls back to a non‑OpenMP build if the toolchain is missing, but we **recommend** ensuring OpenMP is available for large datasets.

---

## Citing
If this pipeline contributes to your research, please cite both foundational papers:

- **[Brandt et al. 2017](https://ui.adsabs.harvard.edu/abs/2017JATIS...3d8002B/abstract)**, *JATIS* 3, 4, 8002  
  DOI: 10.1117/1.JATIS.3.4.048002
- **[Samland et al. 2022](https://ui.adsabs.harvard.edu/abs/2022A%26A...668A..84S/abstract)**, *A&A* 675, A13  
  DOI: 10.1051/0004-6361/202346758

The ADS entries are linked in the bibliography below.

---

## Contributing
We welcome issues & PRs—large or small.

1. **Fork** → **create a feature branch** → **commit** → **open a PR**.  
2. Run `pre-commit run --all-files` to satisfy lint & formatting hooks.  
3. New here? Open an issue or email [Tim Brandt](mailto:timothy.d.brandt@gmail.com) or [Matthias Samland](mailto:matthias.samland@gmail.com) for guidance.

---

## License
This project is distributed under the **BSD-3-Clause License**—see [`LICENSE`](LICENSE) for the full text.

---

### Bibliography
* [Brandt, T. D., *et al.* 2017](https://ui.adsabs.harvard.edu/abs/2017JATIS...3d8002B/abstract), "CHARIS Data Reduction Pipeline", **JATIS**, 3, 048002  
* [Samland, M., *et al.* 2022](https://ui.adsabs.harvard.edu/abs/2022A%26A...668A..84S/abstract), "A New SPHERE IFS Pipeline Based on CHARIS", **A&A**, 675, A13
