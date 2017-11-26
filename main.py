import cv2
from skimage import io
from code import interact

from matplotlib.pyplot import figure, subplot
from skimage import data as img
from skimage import filters
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import closing
from skimage.segmentation import clear_border
import skimage.morphology as mp
import numpy as np
from scipy import ndimage as ndi
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from ipykernel.pylab.backend_inline import flush_figures
from skimage.measure import find_contours
from skimage import feature
import skimage
import os
import matplotlib.patches as mpatches
import cv2
from glob import glob


import PIL
from IPython.display import display, Image

def getMean(desc, image):
    mean = np.mean(image)
    print("{} image mean = {}".format(desc,mean))
    return mean

def getMax(desc, image):
    maxV = max(np.max(image, axis=1))
    print("{} image max = {}".format(desc, maxV))
    return maxV

def neighbour(array, x, y):
    for a in range (od,do):
        for b in range(od,do):
            if array[x+a][y+b] !=0 and array[x+a][y+b] !=255:
                array[x][y] = array[x+a][y+b]
                neighbour2(array,x,y)
                return
    array[x][y] = colors.pop()
    neighbour2(array, x, y)



def neighbour2(array, x, y):
    for a in range(od,do):
        for b in range(od, do):
            if array[x + a][y + b] == 255:
                neighbour(array, x+a, y+b)
                return
    return


def colorize(array):
    rows = len(array)
    columns = len(array[0])
    global colors
   # colors = [10,40,50,60,80,120,140,160,180,200,220,240]
    colors = list(range(1,254,2))
    print(colors)
    for x in range(rows):
        for y in range(columns):
            if array[x][y] == 255:
                neighbour(array, x, y)



def displaySaveImage(imgs, filename="planes_bin.png"):
    fig = figure(figsize=(20,20))
    if len(imgs) == 1:
        rows = 1
    else:
        rows = int(len(imgs)/2 +1)
    for i in range(0, len(imgs)):
        subplot(rows, 2, i+1)
        io.imshow(imgs[i])
    fig.savefig(filename, dpi=500)



def thresh(t):

    binary = label(edges)
    fig, ax = plt.subplots(figsize=(20, 20))

    for region in regionprops(binary,intensity_image=binary):
        # take regions with large enough areas
        if region.area >= 1300:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox

            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
         #   print(region.mean_intensity)
            coin = image[minr:maxr, minc:maxc]
            coinRGB = image2[minr:maxr, minc:maxc]
            coinhsv = skimage.color.rgb2hsv(coinRGB)
            ax.add_patch(rect)
            ax.text(maxc,minr,str((np.mean(coinhsv)))+" ,"+str(np.mean(coinhsv[2])),fontsize=15)



    ax.imshow(binary)
    ax.set_axis_off()
    plt.tight_layout()
    # print(contours)


    #for list in contours:
     #   xs, ys = [*zip(*list)]
        # plt.plot(xs,ys)


        #  print(contours)
   # flush_figures()




directory = os.getcwd()+"\moje" + '/'
images = []

for file in os.listdir(directory):
    image2 = img.load(directory+file,False)
    image = skimage.color.rgb2gray(image2)
    gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    removeNoises = cv2.bilateralFilter(gray,9,75,75)
    hist = cv2.equalizeHist(removeNoises)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    morph = cv2.morphologyEx(hist,cv2.MORPH_OPEN,rectKernel,iterations=15)
   # morph2 =  cv2.morphologyEx(hist,cv2.MORPH_TOPHAT,rectKernel)
    fig, ax = plt.subplots()
    submorph = cv2.subtract(hist, morph)
    thresh = cv2.adaptiveThreshold(submorph, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 10)
    ret, thresh2 = cv2.threshold(submorph,0,255,cv2.THRESH_OTSU)

    canny = cv2.Canny(thresh2,250,255)
    edges = skimage.morphology.dilation(canny)

    # Finding Contours in the image based on edges
    new, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    # Sort the contours based on area ,so that the number plate will be in top 10 contours
    screenCnt = None
    # loop over our contours
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)  # Approximating with 6% error
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:  # Select the contour with 4 corners
            screenCnt = approx
            break
    final = cv2.drawContours(image2, [screenCnt], -1, (0, 255, 0), 3)


    ax.imshow(final)



  #   edges = cv2.Canny(image2,100,200)
  #   rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,6))
  #   edges = cv2.morphologyEx(edges,cv2.MORPH_TOPHAT,rectKernel)
  #   edges = cv2.adaptiveThreshold(edges,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,5,10)
  # #  edges = skimage.feature.canny(edges,1)
  #   edges =skimage.morphology.erosion(skimage.morphology.dilation(edges))
  #   contours = skimage.measure.find_contours(edges,1.2)
  #   fig, ax = plt.subplots()
  #   ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
  #
  #   for n, contour in enumerate(contours):
  #       ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
  #  # ax.imshow(edges)
  #  # meanV = getMean("sobel_max_", edges)
  #   #thresh(0.08)
plt.show()
