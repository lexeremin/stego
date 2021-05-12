import cv2
import numpy as np
import pywt
from pathlib import Path
import matplotlib.pyplot as plt

def plotter(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def main():

    #0. Load stegocontainer(cover image) with shape 512x512 and set as grayscale
    imgPath = 'path/to/stegocontainer.jpg'
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    plotter(img)

    #1. Apply 2d discrete wavelet transformation to get LL, LH, HL, HH filter with shape 256x256
    coeffs2 = pywt.wavedec2(img, "haar")
    # LL = coeffs2[-4]
    (LH,HL,HH) = coeffs2[-2]
    # wavelet_plotter(LL, LH, HL, HH)
    print('Wavelet coefficient shape: ', LH.shape)

    #2. Calculate average for each pass-filter(subband) 
    # tbh useless step(see 5-8 inner loop)
    avgLH = np.mean(LH)
    avgHL = np.mean(HL)
    avgHH = np.mean(HH)
    imgLH = HH
    plotter(imgLH)
    print('AvgLH: ', avgLH, ' AvgHL: ', avgHL, ' AvgHH: ', avgHH)

    #3. Set subband order by its avarage values 
    # tbh useless step(see 5-8 inner loop)
    order = {'LH': avgLH, 'HL': avgHL, 'HH': avgHH}
    order = dict(sorted(order.items(), key=lambda x: x[1]))
    order['LH'] = LH
    order['HL'] = HL 
    order['HH'] = HH 

    #4. Locate position of the subband to be replaced with watermark bits
    a = 0
    n = 3
    zm = 255
    au = lambda j : a + np.floor(1.5*(j+1))
    z = lambda j : zm - np.floor(5.5*(j-1))
    d = lambda j : np.floor(z(j)-au(j))/(n-1)
    an = lambda j : au(j)+np.floor((j-1)*d(j))

    #premade array to load
    parts = np.zeros((3,3,256))
    print(parts[0].shape)

    #5-8. Putting bits of watermark into the cover image
    for p, part in enumerate(parts):
        for k, v in order.items():
            for j in range(256):
                for i in range(3):
                    # np.floor(j/(3-i)) - helps balancing distibution and 
                    # 50*p - offset for + 50 position for each part so they won't overlap
                    pos = int((au(np.floor(j/(3-i)))+(j%8-1)*d(np.floor(j/(3-i)))+50*p)%256) 
                    # part[i][j] = order[k][j][pos]
                    #HH works for better hiding but its easier to corrupt
                    #its useless to put watermark in other subbands
                    part[i][j] = order['HH'][j][pos]

    #9. Reshape watermark to 48x48
    mes = []
    for part in parts:
        mes.append(np.ravel(part))
    mes = np.ravel(mes)
    mes = np.reshape(mes, (48, 48))

    #hardcodded noise reduction filter
    avg = np.mean(mes)
    print("AVG:", avg)
    mes[mes>=0.7*avg] = 255
    mes[mes<0.7*avg] = 0

    plotter(mes)
    cv2.imwrite('path/to/save.jpg', mes)



if __name__ == "__main__":
    main()