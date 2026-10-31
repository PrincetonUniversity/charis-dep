"""Link a checkout's calibration tree into a CHARIS_CALIBRATION_DIR layout, so tests never download.

Usage: python link_calibrations.py <checkout>/calibrations <target>

Creates <target>/<version>/<instrument> -> <checkout>/calibrations/<instrument> for every
instrument in the registry, with the version pinned there.
"""

import json
import sys
from pathlib import Path

source, target = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
registry_path = source.parent / "src" / "charis" / "calibration_registry.json"
registry = json.loads(registry_path.read_text())

for instrument, entry in registry.items():
    link = target / entry["version"] / instrument
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.is_symlink():
        link.symlink_to(source / instrument)
    print(f"{link} -> {source / instrument}")
