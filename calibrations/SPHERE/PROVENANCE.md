# SPHERE IFS calibration data: provenance

Calibration data version 2026.10.0 for SPHERE IFS, the data shipped inside the charis package up to 2.2.0.
None of the files carries provenance header keywords; this table is based on the charis git
history and on the paper describing the SPHERE support:

- **S22**: Samland et al. 2022, A&A 668, A84, doi:[10.1051/0004-6361/202244587](https://doi.org/10.1051/0004-6361/202244587)

Commits refer to the charis repository, <https://github.com/PrincetonUniversity/charis-dep>.

Status:
`reproducible` means the generator is in charis and regenerates the file byte-identically;
`code in charis` means the generator exists in charis but the output was not re-verified;
`not reproducible` means the script or its inputs were not recorded.

| File | What it is | Author, date (commit) | Method and generator | Status |
| --- | --- | --- | --- | --- |
| `mask.fits` | Detector bad-pixel mask (4.64 % bad pixels) | M. Samland, 2021-04-27 (4cd2a5b); dtype reduced 2021-10-11 (7cd2a72) | Script not recorded | not reproducible |
| `pixelflat.fits` | Detector pixel flat | M. Samland, 2020-02-21 (911f650) | Script not recorded | not reproducible |
| `background_template.fits` | Mean and first 3 principal components of the internal background | T. Brandt, 2021-08-10 (0e9231a) | PCA of 234 YH internal background frames from 2015 to 2021 (S22 §2.4); script not in charis | not reproducible |
| `background_scaling_mask.fits` | Pixels used to fit the background scaling | T. Brandt, M. Samland, 2021-10-11 (7cd2a72) | Brandt's mask from the background PCA (S22 §2.4), with 48 further pixels excluded that are bad in `mask.fits` | not reproducible |
| `hexagon_mapping_calibration.json` | Overlap of the hexagonal lenslet grid with the square output grid | M. Samland, 2018-08-19 (012d830), 2018-08-20 (2664b5f) | `charis.image.image_geometry.make_mapping(201)`; fails on NumPy 2 | code in charis |
| `neighbour_indices.json` | Neighbouring lenslets of each lenslet, without the centre | M. Samland, 2018-08-20 (2664b5f) | `charis.image.image_geometry.make_calibration_neighbour_indices(201, include_center=False)` | reproducible |
| `YJ/hires_psflets_lam*.fits`, `YH/hires_psflets_lam*.fits` | Oversampled PSFlet templates at the laser wavelengths | T. Brandt, 2019-11-20 (d58958a, 9860e87) | charis `buildcalibrations`: oversampled PSFlet fit to dedicated single-laser IFS calibration frames, diffuse background subtracted (S22 §2.5–2.6); the input frame set and the options were not recorded | not reproducible |
| `YJ/lamsol.dat`, `YH/lamsol.dat` | Wavelength solution belonging to the PSFlet templates | T. Brandt, 2019-11-20 (d58958a, 9860e87) | charis `buildcalibrations`, from the same fit as the PSFlet templates (S22 §2.5–2.6); the input frame set and the options were not recorded | not reproducible |
| `YH/lensletflat.fits` | Lenslet flat | M. Samland, 2020-02-24 (401eb78) | From a SPECPOS lamp cube of 2017-11-30 extracted with charis; script not in charis | not reproducible |
| `YJ/lensletflat.fits` | Lenslet flat | M. Samland, 2021-05-03 (109f795) | Copy of `YH/lensletflat.fits` | not reproducible |
| `YH/YH_tottrans.dat` | Total transmission, YH mode | M. Samland, 2018-09-04 (17515a1) | Instrument throughput times Paranal atmospheric transmission; script and inputs not in charis | not reproducible |
| `YJ/YJ_tottrans.dat` | Total transmission, YJ mode | M. Samland, T. Brandt, 2018-04-05 (a65e21b) | As for YH, with the YJ throughput; script and inputs not in charis | not reproducible |
