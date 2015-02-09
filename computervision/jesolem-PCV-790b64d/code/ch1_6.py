# -*- coding: utf-8 -*-
# Gaussian blurring & scipy
from PIL import Image
from numpy import *
from pylab import *
from scipy.ndimage import filters

fname = "C:\\skunkworx\\Area52\\home\\6008895\dev\\bitbucket\\me-ml\\computervision\\jesolem-PCV-790b64d\\data\\empire.jpg"
sigs = [2,3,5]

im = array(Image.open(fname).convert("L"))

figure()    

subplot(1,4,1)
xlabel("original")
imshow(im)

for i in range(len(sigs)):
    im2 = zeros(im.shape)
    im2 = filters.gaussian_filter(im, sigs[i])
    im2 = uint(im2)
    subplot(1,4,2+i)
    xlabel(r'$\sigma$=%d' % sigs[i])
    imshow(im2)

suptitle("Gaussian Blur", fontsize=18)             
show()