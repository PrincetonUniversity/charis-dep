#!/usr/bin/env python

import glob
import multiprocessing
import os
import sys
from builtins import input, range, str

from astropy.io import fits
from charis import buildcalibrations, instruments

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Must call buildcal with at least one argument:")
        print("  The path to the narrow-band flatfield image")
        print("Example: buildcal CRSA00000000.fits")
        print("Optional additional arguments: filenames of darks")
        print("  taken with the same observing setup.")
        print("Example: buildcal CRSA00000000.fits darks/CRSA*.fits")
        exit()

    infile = sys.argv[1]
    bgfiles = []
    bgimages = []
    for i in range(2, len(sys.argv)):
        bgfiles += glob.glob(sys.argv[i])
    print(f"Background files: {bgfiles}")
    header = fits.open(infile)[0].header
    instrument, calibration_wavelength, correct_header = \
        instruments.instrument_from_data(header, interactive=True, verbose=True)

    print("\n" + "*" * 60)
    print("Oversample PSFlet templates to enable fitting a subpixel offset in cube")
    print("extraction?  Cost is a factor of ~2-4 in the time to build calibrations.")
    print("*" * 60)
    while True:
        upsample = input("     Oversample? [Y/n]: ")
        if upsample in ['', 'y', 'Y']:
            upsample = True
            break
        elif upsample in ['n', 'N']:
            upsample = False
            break
        else:
            print("Invalid input.")

    ncpus = multiprocessing.cpu_count()
    print("\n" + "*" * 60)
    print("How many threads would you like to use?  %d threads detected." % (ncpus))
    print("*" * 60)
    while True:
        nthreads = input("     Number of threads to use [%d]: " % (ncpus))
        try:
            nthreads = int(nthreads)
            if nthreads < 0 or nthreads > ncpus:
                print("Must use between 1 and %d threads." % (ncpus))
            else:
                break
        except:
            if nthreads == '':
                nthreads = ncpus
                break
            print("Invalid input.")

    print("\n" + "*" * 60)
    print("Building calibration files, placing results in current directory:")
    print(os.path.abspath('.'))
    print("\nSettings:\n")
    print("Using %d threads" % (nthreads))
    print("Narrow-band flatfield image: " + infile)
    print("Wavelength:", calibration_wavelength)
    print("Observing mode: " + instrument.observing_mode)
    print("Upsample PSFlet templates? ", upsample)
    if len(bgfiles) > 0:
        print("Background count rates will be computed.")
    else:
        print("No background will be computed.")
    print("*" * 60)
    while True:
        do_calib = input("     Continue with these settings? [Y/n]: ")
        if do_calib in ['', 'y', 'Y']:
            break
        elif do_calib in ['n', 'N']:
            exit()
        else:
            print("Invalid input.")

    if instrument.instrument_name == 'CHARIS':
        if calibration_wavelength <= instrument.wavelength_range[0] \
                or calibration_wavelength >= instrument.wavelength_range[1]:
            raise ValueError("Error: wavelength " + str(calibration_wavelength) + " outside range (" +
                             str(instrument.wavelength_range[0])
                             + ", " + str(instrument.wavelength_range[1]))
            # + ") of mode " + band)

    inImage, hdr = buildcalibrations.read_in_file(
        infile, instrument, calibration_wavelength,
        ncpus=nthreads, bgfiles=bgfiles)

    buildcalibrations.buildcalibrations(
        inImage, instrument,
        calibration_wavelength.value, mask=None,
        upsample=upsample,
        order=None, header=hdr,
        ncpus=nthreads, verbose=True)
