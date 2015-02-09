from PIL import Image
from pylab import *

fname = "C:\\skunkworx\\Area52\\home\\6008895\dev\\bitbucket\\me-ml\\computervision\\jesolem-PCV-790b64d\\data\\empire.jpg"
im = array(Image.open(fname))
print im.shape, im.dtype

im = array(Image.open(fname).convert("L"))
print im.shape, im.dtype