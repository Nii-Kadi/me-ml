# -*- coding: utf-8 -*-
# Image derivatives & Gaussian derivative filters
from PIL import Image
from numpy import *
from pylab import *
from scipy.ndimage import filters

fname = "C:\\skunkworx\\Area52\\home\\6008895\dev\\bitbucket\\me-ml\\computervision\\jesolem-PCV-790b64d\\data\\empire.jpg"
sigma = 5

im = array(Image.open(fname).convert("L"))

figure()    

subplot(1,4,1)
xlabel("orig. grayscale")
imshow(im)

imx = zeros(im.shape)
filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
subplot(1,4,2)
xlabel(r'x-derivative, $\sigma$=5')
imshow(imx)

imy = zeros(im.shape)
filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)
subplot(1,4,3)
xlabel(r'y-derivative, $\sigma$=5')
imshow(imy)

magnitude = sqrt(imx**2 + imy**2)
subplot(1,4,4)
xlabel("gradient magnitude")
imshow(magnitude)

suptitle("Gaussian Derivative Filters", fontsize=18)             
show()