import cv2
import numpy as np
import pywt
from pathlib import Path
import matplotlib.pyplot as plt

#Plot images
def plotter(img):
    plt.imshow(img, cmap='gray')
    plt.show()

# 1st Approach. Warping chages of position of SP and its neighbornhoods
def warpingEmbedding(I, W, cHarris):
    ...

#2nd Approach. Saliency Point Embedding
def SPE(cHarris, dI, w, d):
    for i in range(cHarris.shape[0]):
        for j in range(cHarris.shape[1]):
            if cHarris[i][j] == 255 and dI[i][j] == 0:
                if w == 255:
                    # dI[i][j] = dI[i][j]+d*dI[i][j]
                    dI[i][j] = dI[i][j]+20
                else:
                    # dI[i][j] = dI[i][j]-d*dI[i][j]  
                    dI[i][j] = dI[i][j]-20      

def main():
    
    #1. Load stegocontainer(cover image) with shape 512x512 and set as grayscale
    stegofn = 'path/to/stegocontainer.jpg'
    img = cv2.imread(stegofn)
    I = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    plotter(img)

    
    #3. Load stegomessage(watermark image) with shape 48x48 and set as grayscale
    wimg = 'path/to/wtmrk.jpg'
    W = cv2.imread(wimg)
    W = cv2.cvtColor(W,cv2.COLOR_BGR2GRAY)
    # plotter(mes)
    #set watermark as binary image
    # mes[mes>127] = 255
    # mes[mes<=127] = 0
    plotter(W)
    print('stegomessage(img) shape: ', W.shape)
    # plotter(mes)

    #3. Harris corner detection function
    I = np.float32(I)
    #block size 3x3, ksize 3x3, detector free parameter 0,04
    cHarris = cv2.cornerHarris(I,3,3,0.04)
    #result is dilated for marking the corners, not important
    cHarris = cv2.dilate(cHarris,None)
    plotter(cHarris)
    print(len(np.unique(cHarris)))
    print('Ratio char. pixels: ', len(W)**2/len(cHarris[cHarris>0.01*cHarris.max()]))
    # Threshold for an optimal value, it may vary depending on the image.
    dI = np.zeros(I.shape)
    # dI[cHarris>0.01*cHarris.max()]=255
    cHarris[cHarris>0.01*cHarris.max()] = 255
    cHarris[cHarris<=0.01*cHarris.max()] = 0
    
    # plotter(I)

    W[W>127] = 255
    W[W<=127] = 0

    for w in W:
        SPE(cHarris, I, w, 0.1)

    # I = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY, 0, 255)
    # Iw = I+dI

    #4. Reshape binary watermark image to 1x2304 array first then to 3x768 array
    # mes = np.ravel(mes)
    # mes = np.reshape(mes, (3, int(mes.shape[0]/3)))
    # print(mes.shape)
    #13. Save stegocontainer(modded cover image)
    plotter(I)
    cv2.imwrite('path/to/save.jpg', I)


if __name__ == "__main__":
    main()