# -*- coding: utf-8 -*-
# Histogram equalization
from PIL import Image
from numpy import *
from pylab import *

fname = "C:\\skunkworx\\Area52\\home\\6008895\dev\\bitbucket\\me-ml\\computervision\\jesolem-PCV-790b64d\\data\\empire.jpg"

def histeq(im, nbins=256):
    #Histogram equalization of a grayscale image
    
    # get image histogram
    imhist,bins = histogram(im.flatten(), nbins, density=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
    
    # use linear interpolation of cdf to find new pixel values
    im2 = interp(im.flatten(), bins[:-1], cdf)
    
    return im2.reshape(im.shape), cdf


im = array(Image.open(fname).convert("L"))
# !!! 
im2, cdf = histeq(im) 

figure()    
subplot(2,3,4)
axis("equal")
axis("off")
imshow(im)

subplot(2,3,1)
xlabel("before")
hist(im.flatten(), 128)
 
subplot(2,3,2)
xlabel("transform")
plot(cdf)

subplot(2,3,6)
axis("equal")
axis("off")
imshow(im2)

subplot(2,3,3)
xlabel("after")
hist(im2.flatten(), 128)
 
show()