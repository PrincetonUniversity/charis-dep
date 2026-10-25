"""Download the static calibration data, or show where it is kept."""

import argparse
import logging

from charis.calibration_data import REGISTRY, cache_root, calibration_root


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="charis-calibrations",
        description="Manage the static calibration data that charis downloads from Zenodo.")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser(
        "fetch", help="download the calibration data of an instrument if it is not there yet")
    fetch.add_argument("instrument", choices=[*REGISTRY, "all"])
    commands.add_parser("path", help="print the calibration root directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "path":
        print(cache_root())
        return
    instruments = list(REGISTRY) if args.instrument == "all" else [args.instrument]
    for instrument in instruments:
        print(calibration_root(instrument) / instrument)


if __name__ == "__main__":
    main()
