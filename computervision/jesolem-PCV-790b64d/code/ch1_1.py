from PIL import Image
from pylab import *

fname = "C:\\skunkworx\\Area52\\home\\6008895\dev\\bitbucket\\me-ml\\computervision\\jesolem-PCV-790b64d\\data\\empire.jpg"
im = array(Image.open(fname))

imshow(im)

x = [100,100,400,400]
y = [200,500,200,500]

plot(x, y, "ks:")
plot(x[:2], y[:2])

title("empire.jpg")
show()