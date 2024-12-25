import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2



file_photo="./desktop-wallpaper-full-nature-04-jpeg-1600×900-paysage-coucher-de-soleil-fond-ecran-paysage-nature-paysage.jpg"

water_mark="./fond-ecran-windows-seven-wallpaper-32.jpeg"

image =cv2.imread(file_photo,cv2.IMREAD_GRAYSCALE)

image_water_mark=cv2.imread(water_mark,cv2.IMREAD_GRAYSCALE)


image_water_mark=cv2.resize(image_water_mark,(image.shape[1],image.shape[0]))



c = pywt.wavedec2(image, 'db2', mode='periodization',level=3)


c_water_mark=pywt.wavedec2(image_water_mark, 'db2', mode='periodization',level=3)


#arr, slices = pywt.coeffs_to_array(c)

#plt.figure()

#plt.imshow(arr,cmap=plt.cm.gray)

#plt.show()

#print(slices)

LL3, (HL3, LH3, HH3), (HL2, LH2, HH2), (LL1, LH1, HH1) = c

WLL3, (WHL3, WLH3, WHH3), (WHL2, WLH2, WHH2), (WLL1, WLH1, WHH1)=c_water_mark



LL3=0.85*LL3+0.009*WLL3

c_reconstru=[LL3, (HL3, LH3, HH3), (HL2, LH2, HH2), (LL1, LH1, HH1)]



img_reconstru=pywt.waverec2(c_reconstru,'db2',mode='periodization')




plt.figure()
plt.imshow(img_reconstru,cmap=plt.cm.gray)
plt.show()


