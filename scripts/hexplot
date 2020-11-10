#!/usr/bin/env python

import os
import subprocess
import sys

import pkg_resources

filename = sys.argv[1]

cwd = os.getcwd()
script_path = pkg_resources.resource_filename('charis', 'hexplot.py')

subprocess.run("bokeh serve --show {} --args '{}'".format(script_path, filename),
               shell=True, check=True, stdout=subprocess.PIPE)
