#!/usr/bin/env python

import tools
import unittest
import configFiles as configs


def reduceCHARIS(config):
    """
    Function reduceCHARIS holds the main steps in the data reduction.
    It requires a configuration set as input.
    """  
    



if __name__ == '__main__':

    test_loader = unittest.TestLoader()
    tests = test_loader.discover('image', 'test*.py')
    test_runner = unittest.runner.TextTestRunner()
    test_runner.run(tests)
    
    reduceCHARIS(configs)
