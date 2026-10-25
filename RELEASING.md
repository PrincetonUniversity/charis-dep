# Releasing charis

charis is published on PyPI as [`charis`](https://pypi.org/p/charis) by
[`.github/workflows/publish.yml`](.github/workflows/publish.yml).
The workflow builds an sdist and wheels for CPython 3.11–3.13 on Linux (x86_64, aarch64) and
macOS (arm64, x86_64) with [cibuildwheel](https://cibuildwheel.pypa.io) (settings in
`pyproject.toml`), and runs the test suite against every wheel.
It uploads with Trusted Publishing: PyPI trusts that workflow file together with a GitHub
environment, so no API token is stored anywhere.

## One-time setup

These steps need a PyPI account that will own the project and a repository admin.

1. On [pypi.org](https://pypi.org/manage/account/publishing/), add a *pending publisher* for the
   project `charis`: owner `PrincetonUniversity`, repository `charis-dep`, workflow
   `publish.yml`, environment `pypi`. The first upload then creates the project under that
   account; add the other maintainers as owners afterwards.
2. Do the same on [test.pypi.org](https://test.pypi.org/manage/account/publishing/) with the
   environment `testpypi`.
3. In the repository settings (Environments), create `pypi` with *required reviewers* (the
   maintainers who may approve a release) and a deployment rule that allows tags only.
   Create `testpypi` without protection rules.

## Steps

1. On `devel`, rename the CHANGELOG's `[Unreleased]` heading to the new version and date, for
   example `## [2.3.0] — 2026-11-02`, and merge that by PR. CI on `devel` must be green.
   If the calibration data changed, the new Zenodo version and `calibration_registry.json` must
   be in that PR (see [`calibrations/README.md`](calibrations/README.md)).
2. Optional dry run: run the Publish workflow by hand on `devel`
   (`gh workflow run publish.yml --ref devel`), which uploads a dev version to TestPyPI.
   Install it without taking dependencies from TestPyPI:
   `pip download --no-deps --index-url https://test.pypi.org/simple/ charis==<version>`,
   then `pip install` the downloaded file, so its dependencies come from PyPI.
3. Merge `devel` into `main` by PR.
4. On GitHub, draft a new release: a new tag `2.3.0` (no `v`, like the earlier tags) targeting
   `main`, title `2.3.0`, and the CHANGELOG section as notes.
   Publishing the release starts the Publish workflow.
5. The workflow checks that the version matches the tag, builds and tests the sdist and all
   wheels, and then waits for approval of the `pypi` environment.
   Approve it on the workflow run's page (Actions → Publish → *Review deployments*).
6. In a clean environment, `pip install charis==2.3.0`, check
   `python -c "import charis; print(charis.__version__, charis.openmp_enabled())"`, and extract
   one SPHERE cube (for example `testdata/SPHERE/YH`) to check the calibration download.
7. Merge `main` back into `devel`, so that development versions count from the new tag.
8. Update the packages that depend on charis, such as spherical's `pipeline` extra.

## Things to know

- A version can be uploaded only once, even after deleting it from PyPI.
  If a release is broken, fix it with a new patch version.
- A GitHub release marked as a pre-release (for example `2.3.0rc1`) is published to PyPI as
  well; pip installs it only when asked for pre-releases.
- Renaming `publish.yml` or the environments `pypi` and `testpypi`, or moving the repository,
  breaks publishing until the trusted publishers on pypi.org and test.pypi.org are updated.
- The macOS wheels carry Homebrew's `libomp`, which is built for the runner's macOS version,
  so they require at least that version (the workflow log prints it). Older macOS versions
  build from the sdist, which needs `libomp` (README, section "OpenMP").
- The calibration data is not in the sdist or the wheels; charis downloads it from Zenodo.
