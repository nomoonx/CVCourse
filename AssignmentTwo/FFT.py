import cv2
import numpy as np
from matplotlib import pyplot as plt

image_name=raw_input('Please input the name of the file:')

img = cv2.imread(image_name,0)
percentage=float(raw_input('input the percentage of low frequencies (a number between 0 and 1):'))

rows,cols = img.shape
# to optimize the dft
nrows = cv2.getOptimalDFTSize(rows)
ncols = cv2.getOptimalDFTSize(cols)

nimg = np.zeros((nrows,ncols))
nimg[:rows,:cols] = img

dft = cv2.dft(np.float32(nimg),flags = cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.subplot(222),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum for original FFT'), plt.xticks([]), plt.yticks([])

# set the high frequency to zero
crow,ccol = nrows/2 , ncols/2
mask = np.zeros((nrows,ncols,2),np.uint8)
mask[crow-nrows*percentage/2:crow+nrows*percentage/2, ccol-ncols*percentage/2:ccol+ncols*percentage/2] = 1
fshift = dft_shift*mask


f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
magnitude_spectrum = 20*np.log(cv2.magnitude(fshift[:,:,0],fshift[:,:,1]))

plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum for masked FFT'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(img_back, cmap = 'gray')
plt.title('Output Image'), plt.xticks([]), plt.yticks([])

plt.show()
