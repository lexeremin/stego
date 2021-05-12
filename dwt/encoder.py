import cv2
import numpy as np
import pywt
from pathlib import Path
import matplotlib.pyplot as plt



# only usable with cv2.dwt2() but not with cv2.wavedec2()
def wavelet_plotter(LL, LH, HL, HH, gray = 'gray_r', tLL = 'LL', tLH = 'LH', tHL = 'HL', tHH= 'HH'):
    fig, axes = plt.subplots(nrows = 2, ncols =2 )

    axes[0,0].set(title=tLL)
    axes[0,1].set(title=tLH)
    axes[1,0].set(title=tHL)
    axes[1,1].set(title=tHH)

    # axes[0,0].imshow(np.uint8(LL), cmap='gray')
    # axes[0,1].imshow(np.uint8(LH), cmap='gray')
    # axes[1,0].imshow(np.uint8(HL), cmap='gray')
    # axes[1,1].imshow(np.uint8(HH), cmap='gray')
    axes[0,0].imshow(LL, cmap=gray)
    axes[0,1].imshow(LH, cmap=gray)
    axes[1,0].imshow(HL, cmap=gray)
    axes[1,1].imshow(HH, cmap=gray)
    
    plt.show()

def plotter(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def main():
    
    #1. Load stegocontainer(cover image) with shape 512x512 and set as grayscale
    stegofn = 'path/to/stegocontainer.jpg'
    i = cv2.imread(stegofn)
    img = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
    # plotter(img)

    #2. Apply 2d discrete wavelet transformation to get LL, LH, HL, HH filter with shape 256x256
    coeffs2 = pywt.wavedec2(img, "haar")
    # LL = coeffs2[-4]
    (LH,HL,HH) = coeffs2[-2]
    # coeffs2[-2] = tuple([np.zeros_like(v) for v in coeffs2[-2]])
    # wavelet_plotter(LL, LH, HL, HH)
    print('Wavelet coefficient shape: ', LH.shape)

    #3. Load stegomessage(watermark image) with shape 48x48 and set as grayscale
    messageFN = 'path/to/wtmrk.jpg'
    m = cv2.imread(messageFN)
    mes = cv2.cvtColor(m,cv2.COLOR_BGR2GRAY)
    # plotter(mes)
    #set watermark as binary image
    mes[mes>127] = 128
    mes[mes<=127] = 50
    plotter(mes)
    print('stegomessage(img) shape: ', mes.shape)
    # plotter(mes)

    #4. Reshape binary watermark image to 1x2304 array first then to 3x768 array
    mes = np.ravel(mes)
    mes = np.reshape(mes, (3, int(mes.shape[0]/3)))
    print(mes.shape)

    #5. Calculate average for each pass-filter(subband)
    avgLH = np.mean(LH)
    avgHL = np.mean(HL)
    avgHH = np.mean(HH)
    print('AvgLH: ', avgLH, ' AvgHL: ', avgHL, ' AvgHH: ', avgHH)

    #6. Set subband order by its avarage values
    order = {'LH': avgLH, 'HL': avgHL, 'HH': avgHH}
    order = dict(sorted(order.items(), key=lambda x: x[1]))
    order['LH'] = LH
    order['HL'] = HL
    order['HH'] = HH
    # order['LH'] = np.zeros((256,256))
    # order['HL'] = np.zeros((256,256))
    # order['HH'] = np.zeros((256,256))

    #7. Reshape each part of watermark(3x768) into 3 parts of 3x256:
    parts = []
    for i in range(len(mes)):
        parts.append(np.reshape(mes[i], (3, int(mes[i].shape[0]/3))))
    print(parts[0].shape)

    #8. Locate position of the subband to be replaced with watermark bits
    a = 0
    n = 3
    zm = 255
    au = lambda j : a + np.floor(1.5*(j+1))
    z = lambda j : zm - np.floor(5.5*(j-1))
    d = lambda j : np.floor(z(j)-au(j))/(n-1)
    an = lambda j : au(j)+np.floor((j-1)*d(j))

    #9-11. Putting bits of watermark into the cover image
    for p, part in enumerate(parts):
        for k, v in order.items():
            for j in range(256):
                for i in range(3):
                    # np.floor(j/(3-i)) - helps balancing distibution and 
                    # 50*p - offset for + 50 position for each part so they won't overlap
                    pos = int((au(np.floor(j/(3-i)))+(j%8-1)*d(np.floor(j/(3-i)))+50*p)%256)
                    #HH works for better hiding but its easier to corrupt
                    #its useless to put watermark in other subbands
                    order['HH'][j][pos] = part[i][j]

    #12. Apply inverse DWT to modified cover image
    LH = order['LH']
    HL = order['HL']
    HH = order['HH']
    coeffs2[-2] = tuple([LH, HL, HH])
    stegoimg = pywt.waverec2(coeffs2, 'haar')
   
    stegoimg = cv2.GaussianBlur(stegoimg,(0,0),1)
    plotter(stegoimg)
    #13. Save stegocontainer(modded cover image)
    cv2.imwrite('path/to/save.jpg', stegoimg)


if __name__ == "__main__":
    main()