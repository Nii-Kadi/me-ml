from PIL import Image
from pylab import *

fname = "C:\\skunkworx\\Area52\\home\\6008895\dev\\bitbucket\\me-ml\\computervision\\jesolem-PCV-790b64d\\data\\empire.jpg"
im = array(Image.open(fname).convert("L"))

figure()
gray()
contour(im, origin="image")
axis("equal")
axis("off")

figure()
hist(im.flatten(), 128)

show()