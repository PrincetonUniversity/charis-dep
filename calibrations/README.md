# Static calibration data (maintainer notes)

This directory holds the static calibration data of each instrument (`SPHERE/`, `CHARIS/`).
It is tracked in git but is not part of the Python package: charis downloads it from Zenodo on
first use (`src/charis/calibration_data.py`) into
`<root>/<version>/<instrument>/`, where `<root>` is `$CHARIS_CALIBRATION_DIR` or
`${XDG_CACHE_HOME:-~/.cache}/charis/calibrations`.

Each instrument is one Zenodo record, `charis-calibrations-SPHERE` and
`charis-calibrations-CHARIS`. New calibrations become a new version of the same record.
`src/charis/calibration_registry.json` pins, per instrument, the exact set a charis release was
tested with:

- `version`: the data version (`YYYY.MM.N`, e.g. `2026.10.0`). It is both the Zenodo record's
  version field and the name of the cache folder.
- `record_id` and `doi`: of the Zenodo **version** record, never the concept record.
- `archive`: the file name on Zenodo, `charis-calibrations-<instrument>-<version>.tar.gz`.
- `md5`: of that archive.

There is no "latest" lookup: code and data change together, in one commit.

## Building an archive

The archive contains the instrument directory at its top level, including its `PROVENANCE.md`:

```bash
cd calibrations
# COPYFILE_DISABLE keeps macOS tar from adding ._ resource-fork files.
COPYFILE_DISABLE=1 tar --exclude .DS_Store -czf charis-calibrations-SPHERE-2026.10.0.tar.gz SPHERE
md5sum charis-calibrations-SPHERE-2026.10.0.tar.gz      # macOS: md5 -q <archive>
```

Before uploading, extract it into an empty directory `<scratch>/<version>/` and run `getcube` on
real data with `CHARIS_CALIBRATION_DIR=<scratch>`.

## Publishing a new version

1. Change the files here and update the instrument's `PROVENANCE.md` (rules below).
2. Build the archive and get its md5 as above.
3. On Zenodo, open the instrument's record and choose "New version". Upload the archive, set the
   version field to the new data version, and repeat the provenance (recipe and commit) in the
   version notes. The draft shows the new record ID and DOI before it is published.
4. In `src/charis/calibration_registry.json`, set `version`, `record_id`, `doi`, `archive` and
   `md5` for that instrument, and change the data version in the test job of
   `.github/workflows/ci.yml`, which links this directory instead of downloading it.
5. Publish the Zenodo version (permanent), then check that `charis-calibrations fetch <instrument>`
   into an empty `CHARIS_CALIBRATION_DIR` works.

Old versions stay on Zenodo and in users' caches; charis never deletes a cache folder.

## Provenance rule for new versions

Enforced by review, not by code. Every calibration version after 2026.10.0:

1. Commits its generating script under `calibrations/recipes/<instrument>/` in this repository,
   not in a separate repository, because code and data change together. Scripts take paths as
   arguments; nothing is hard-coded.
2. Commits the exact inputs next to the script: the ESO archive file IDs (DP.ID), or the archive
   query plus the random seed. Raw ESO frames are not re-uploaded; they stay in the ESO archive.
3. Names the recipe and the commit it was run from in the file's `PROVENANCE.md` row, and the
   Zenodo version notes repeat it.

The `PROVENANCE.md` files of 2026.10.0 record what is known about the earlier files; most of
their scripts or inputs were not recorded.
