# Changelog

All notable changes to this project are documented in this file.

This project follows [Semantic Versioning](https://semver.org/) and the
[Keep a Changelog](https://keepachangelog.com/) format. Commit hashes are given for changes whose
rationale is too long for a changelog entry; the commit messages carry the measurements.

---

## [Unreleased]

Corrections to the SPHERE/IFS inverse variance and cube-header angles, and an OpenMP build that is parallel on Linux too.
**Re-extract SPHERE cubes before combining them with 2.1.0 products, including cubes reduced through spherical:**
`ivar` is now in counts, larger by about the gain squared (about 3.2),
and the header angles change for targets just south of the equator, for sequences that cross the sidereal-time wrap, and for field-tracking data.

### Added

- A `test` extra (`pytest`, `ruff`) for running the test suite.
- `charis.openmp_enabled()` reports whether the installed build is parallel.
  `CHARIS_REQUIRE_OPENMP=1` makes a missing OpenMP a build error, and `CHARIS_NO_OPENMP=1` builds serially on purpose.

### Changed

- The license metadata now matches the `LICENSE` file (`BSD-2-Clause-Views`).
- `nonlinear_threshold` now defaults to `None` (it was an inert 40000).
  When set, SPHERE pixels above that count level in a DIT are masked as bad pixels, recorded as `NLTHRESH` and `NLNPIX` in the header ([#51](https://github.com/PrincetonUniversity/charis-dep/issues/51)).
- `getcube` and the `extractcube` command default to the instrument gain (1.8 for SPHERE) instead of 2.

### Fixed

- SPHERE `ivar` is the inverse variance in counts, matching the data, and scales correctly with the number of averaged DITs ([#52](https://github.com/PrincetonUniversity/charis-dep/issues/52)).
- The `PARANG`, `PUPIL OFFSET` and `DEROT ANGLE` header values are correct for declinations between 0° and −1°, across the sidereal-time wrap, and in field tracking, which no longer crashes extraction ([#53](https://github.com/PrincetonUniversity/charis-dep/issues/53)).
  spherical computes its own angles, so this part does not affect spherical reductions.
- Linux builds compile with OpenMP, and macOS builds find `libomp` in the active conda or pixi environment or in either Homebrew prefix.
  A build without OpenMP now logs a warning at run time when more than one thread is requested.
- A CHARIS file without the `Y_FLTNAM` or `OBJECT` mode keywords (typical of data before 2017-04-01) raises a clear `ValueError` instead of an `UnboundLocalError`, unless `interactive=True` is passed ([#11](https://github.com/PrincetonUniversity/charis-dep/issues/11)).

---

## [2.1.0] — 2026-07-29

Substantial corrections to the SPHERE/IFS inverse-variance chain, plus a preprocessing-only mode
and a more robust `fitshift`. **Cubes extracted with 2.0.x are not comparable to 2.1.0 cubes**: the
resampled `ivar` scale, the bad-lenslet mask and (where `fitshift` diagnostics failed) the PSFlet
registration have all changed. Re-extract before combining old and new products.

### Added

- **Preprocessing-only mode** — `extractcube(..., preprocessing_only=True)` stops after up-the-ramp
  combination, background subtraction and bad-pixel correction, returning the cleaned 2-D frame
  instead of a cube. Exposed as `preprocessing_only` in the `[Extract]` config section.
- **`fitshift_nchunks`** — the number of chunks the localised PSFlet shift fit uses is now a
  parameter instead of a hardcoded constant (defaults: 16 for CHARIS, 1 for SPHERE).
- **`save_fitshift_diag`** (default `False`) — writes the `fitshift` QC diagnostic to
  `{outdir}/{basename}_fitshift_diag.fits`. Previously this was tied to `saveresid`, which has its
  own residual-image purpose.
- **`resample_ivar_cube` and `resample_good_fraction_cube`** (`charis.image.image_geometry`) —
  variance-propagating resampling and a soft good-area map, the latter written as a
  `*_cube_resampled_goodfrac*.fits` sidecar.
- **A `pixi.toml`** with `test`, `notebook` and `dev` environments and `test` / `lint` / `format`
  tasks.
- **Tests** for the resampling, the `fitshift` diagnostics and the hexagonal flagging.

### Changed

- **The SPHERE hexagonal bad-lenslet mask is one-sided on inverse variance**
  ([`fd35ece`](https://github.com/PrincetonUniversity/charis-dep/commit/fd35ece)). The two-armed
  rule flagged a *different* 4.48 % of in-field spaxel-channels in every frame, most of it sampling
  error in a six-sample `mad_std` rather than defects. It is now a single one-sided pass with a
  scale floor and a minimum-neighbour requirement, masking 2.67 % per frame and stable frame to
  frame, at roughly a third of the runtime. `_smoothandmask_hexgeometry` lost its unused `good`
  argument and gained `out_of_field`.
- **The microlens footprint is taken from the lenslet flat, not inferred from the data.** Padding
  lenslets on the rim of the field carry a small crosstalk flux at some wavelengths and an exact
  zero at others, so a footprint inferred from `data == 0` classified the same lenslet differently
  at different wavelengths: on 51 Eri OBS_H it left 315 lenslets NaN in one contiguous wavelength
  block and interpolated at the rest, putting 300–480 NaN pixels per channel *inside* the
  resampled footprint. `_out_of_field_lenslets` uses the flat's two sentinels instead, and any
  in-field spaxel that still cannot be interpolated falls back to zero flux with `ivar == 0`.
  No non-finite flux now survives inside the field, so a downstream footprint can be inferred
  from the all-NaN border alone. Masking rate and runtime are unchanged.
- **`suppressrn` is force-disabled for SPHERE**, with a warning — the ESO DRS CDS already removes
  channel-correlated read noise, so applying it again degraded SPHERE extractions. The parameter
  default is unchanged for CHARIS ([#36](https://github.com/PrincetonUniversity/charis-dep/issues/36)).
- **The static bad-lenslet definition is shared by both extraction paths** — `fit_spectra`
  previously omitted the `lensletflat < 0.7` term that `optext_spectra` applied.
- **Improved localised `fitshift`**, and expanded `getcube` / `buildcalibration` docstrings.
- **`ivar` is no longer interpolated before being zeroed at bad pixels.** All three sites in
  `extractcube` now zero directly. No change to any delivered array — `sph_ifs_fix_badpix`
  writes only inside the run of bad pixels it repairs, which the following line overwrites —
  but the three sites had drifted apart and the surviving comment described a corruption that
  does not occur ([#50](https://github.com/PrincetonUniversity/charis-dep/issues/50)).

### Removed

- **`charis/test_utr.py`** — not a test but an old manual script, collected only because of its
  name. It called a signature that no longer exists and read a hardcoded path, and its collection
  error aborted the whole suite.
- **Five uncalled `charis.tools.toolbox` functions** and **two unreachable `hexplot` helpers**, one
  of which called a function that exists nowhere in the repository. Checked for external callers
  against [spherical](https://github.com/m-samland/spherical) and
  [trap](https://github.com/m-samland/trap) before removal.

### Fixed

- **Inverse variance is now propagated through the hexagon→square resampling**
  ([`75f2246`](https://github.com/PrincetonUniversity/charis-dep/commit/75f2246),
  [#42](https://github.com/PrincetonUniversity/charis-dep/issues/42)).
  The flux-conserving operator was applied to `ivar` directly, which understated the noise by
  ~16× on SPHERE OBS_H and averaged masked-lenslet zeros away, so `ivar == 0` found nothing
  downstream and bad spaxels did not survive the resample.
- **A failed `fitshift` diagnostic write no longer discards a successful shift fit**
  ([`d37f765`](https://github.com/PrincetonUniversity/charis-dep/commit/d37f765)). The write sat
  inside the `try` guarding the fit, so any I/O error silently reverted to unshifted PSFlets —
  corrupting astrometry for the sake of a never-read-back QC file.
- **Background photon noise is included in `ivar`** after PCA background subtraction
  ([#35](https://github.com/PrincetonUniversity/charis-dep/issues/35)).
- **`sph_ifs_fix_badpix` interpolation no longer corrupts the noise model** of pixels neighbouring
  a bad pixel ([#37](https://github.com/PrincetonUniversity/charis-dep/issues/37)).
- **Masked SPHERE lenslets carry an exact `0`**, not a `1e-15` sentinel, so downstream
  `ivar == 0` tests work ([#40](https://github.com/PrincetonUniversity/charis-dep/issues/40)).
- **`_recalc_ivar` / `_get_corrnoise` take an explicit `channel_width`** instead of assuming one
  ([#38](https://github.com/PrincetonUniversity/charis-dep/issues/38)).

---

## [2.0.1] and earlier

Not tracked in this file. See the commit history and
[Brandt et al. 2017](https://ui.adsabs.harvard.edu/abs/2017JATIS...3d8002B/abstract).
