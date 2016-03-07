import utr
import primitives

fn = 'CRSA00007072.fits'
datadir = '/Users/protostar/Dropbox/data/charis/lab/2016-02-19/'
im = utr.utr_rn(datadir=datadir, filename=fn, refchan=False, return_im=True)
im.filename = 'test'

_x, _y, good, coef = primitives.locatePSFlets(im)

print coef 

print good
