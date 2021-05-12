import cv2
import numpy as np
import pywt
from pathlib import Path
import matplotlib.pyplot as plt

#Plot images
def plotter(img):
    plt.imshow(img, cmap='gray')
    plt.show()

#2nd Approach. Saliency Point Embedding
def detectorSPE(W, cHarris, I):
    ratio = 0
    arrayI = []
    for i in range(cHarris.shape[0]):
        for j in range(cHarris.shape[1]):
            if cHarris[i][j] == 255:
                arrayI.append(I[i][j])
    arrayI = np.array(arrayI)
    if len(arrayI)>len(W):
        arrayI = arrayI[:len(W)]
    else:
        W = W[:len(arrayI)]
    W = W/W.max()
    arrayI = arrayI/arrayI.max()
    for i in range(len(W)):
        ratio+= W[i]/arrayI[i]
    print(ratio/len(W))

def main():

    stegofn = 'path/to/stegocontainer.jpg'
    img = cv2.imread(stegofn)
    I = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # plotter(img)

    
    #3. Load stegomessage(watermark image) with shape 48x48 and set as grayscale
    wimg = 'path/to/wtmrk.jpg'
    W = cv2.imread(wimg)
    W = cv2.cvtColor(W,cv2.COLOR_BGR2GRAY)
    # plotter(mes)
    #set watermark as binary image
    W[W>127] = 255
    W[W<=127] = 0
    cHarris = cv2.cornerHarris(I,3,3,0.04)
    #result is dilated for marking the corners, not important
    cHarris = cv2.dilate(cHarris,None)
    cHarris[cHarris>0.01*cHarris.max()] = 255
    cHarris[cHarris<=0.01*cHarris.max()] = 0
    plotter(W)
    print('stegomessage(img) shape: ', W.shape)
    detectorSPE(W.ravel(), cHarris, I)
    # plotter(mes)

if __name__ == "__main__":
    main()