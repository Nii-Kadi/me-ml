from PIL import Image
from numpy import *
from pylab import *

fname = "C:\\skunkworx\\Area52\\home\\6008895\dev\\bitbucket\\me-ml\\computervision\\jesolem-PCV-790b64d\\data\\empire.jpg"

figure()
im = array(Image.open(fname).convert("L"))
print int(im.min()), int(im.max())
gray()
title("grayscale")
imshow(im)

figure()
im2 = 255 - im
print int(im2.min()), int(im2.max())
title("inverted")
imshow(im2)

figure()
im3 = (100.0/255) * im + 100
print int(im3.min()), int(im3.max())
title("clamped: 100...200")
imshow(im3)

figure()
im4 = 255.0 * (im/255.0)**2
print int(im4.min()), int(im4.max())
title("squared")
imshow(im4)

show()