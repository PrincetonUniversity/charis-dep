import unittest
from image import Image


class ImageTests(unittest.TestCase):
    
    def setUp(self):
        self.testImage = Image('testimage/HICA_withivar.fits')

    def test_is2048x2048(self):
        self.assertEqual(self.testImage.data.shape, (2048, 2048),
                         'Array in last test file HDU is not 2048x2048.')

    def tearDown(self):
        del self.testImage
