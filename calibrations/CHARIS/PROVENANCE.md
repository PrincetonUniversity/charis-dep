# CHARIS calibration data: provenance

Calibration data version 2026.10.0 for CHARIS, the data shipped inside the charis package up to 2.2.0.
None of the files carries provenance header keywords; this table is based on the charis git
history and on the paper describing the pipeline:

- **B17**: Brandt et al. 2017, JATIS 3, 048002, doi:[10.1117/1.JATIS.3.4.048002](https://doi.org/10.1117/1.JATIS.3.4.048002)

Commits refer to the charis repository, <https://github.com/PrincetonUniversity/charis-dep>.
All files are T. Brandt's master calibration files, added on 2016-12-07 (d3321c7, 18b4141) and
moved into one folder per mode (`J`, `H`, `K`, `Broadband`) in 2018 without changes.
No generator scripts are in charis; B17 describes the methods.

Status:
`reproducible` means the generator is in charis and regenerates the file byte-identically;
`code in charis` means the generator exists in charis but the output was not re-verified;
`not reproducible` means the script or its inputs were not recorded.

| File | What it is | Author, date (commit) | Method and generator | Status |
| --- | --- | --- | --- | --- |
| `mask.fits` | Detector bad-pixel mask | T. Brandt, 2016-12-07 (d3321c7, 18b4141); identical copies in every mode folder merged into one file 2021-08-10 (0e9231a) | B17; script not recorded | not reproducible |
| `pixelflat.fits` | Detector pixel flat | T. Brandt, 2016-12-07 (d3321c7, 18b4141); merged as `mask.fits` (0e9231a) | B17; script not recorded | not reproducible |
| `<mode>/hires_psflets_lam*.fits` | Oversampled PSFlet templates at monochromatic calibration wavelengths | T. Brandt, 2016-12-07 (d3321c7, 18b4141) | B17; fit script and inputs not recorded | not reproducible |
| `<mode>/lamsol.dat` | Wavelength solution belonging to the PSFlet templates | T. Brandt, 2016-12-07 (d3321c7, 18b4141) | B17; fit script and inputs not recorded | not reproducible |
| `<mode>/lensletflat.fits` | Lenslet flat | T. Brandt, 2016-12-07 (d3321c7, 18b4141) | B17; script not recorded | not reproducible |
| `<mode>/<mode>_tottrans.dat` | Total transmission of the mode | T. Brandt, 2016-12-07 (d3321c7, 18b4141) | B17; script and inputs not recorded | not reproducible |
