import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2
from scipy import ndimage

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def putText(image, text):
    (h, w) = image.shape[:2]
    org = (w // 2, h // 2)
    frontScale=2.0
    color=(255.0)
    front=cv2.FONT_HERSHEY_SIMPLEX
    thickness=2
    image = cv2.putText(image, text, org, front, frontScale, color, thickness,cv2.LINE_AA)
    return image


file_photo="./desktop-wallpaper-full-nature-04-jpeg-1600×900-paysage-coucher-de-soleil-fond-ecran-paysage-nature-paysage.jpg"

water_mark="./fond-ecran-windows-seven-wallpaper-32.jpeg"

image =cv2.imread(file_photo,cv2.IMREAD_GRAYSCALE)

#image_rotated=ndimage.rotate(image,45)

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

#----------- ici on essaie de camufler le watermark ------------------

LL3_prime=0.85*LL3+0.009*WLL3

c_reconstru=[LL3_prime, (HL3, LH3, HH3), (HL2, LH2, HH2), (LL1, LH1, HH1)]


img_reconstru=pywt.waverec2(c_reconstru,'db2',mode='periodization')

#---ici on essaie de retrouver le watermark---------------
img_reconstru_bis=img_reconstru.copy()

c_img_reconstru_bis=pywt.wavedec2(img_reconstru_bis,'db2',mode='periodization',level=3)

PLL3, (PHL3, PLH3, PHH3), (PHL2, PLH2, PHH2), (PLL1, PLH1, PHH1)=c_img_reconstru_bis

PLL3_prime=(PLL3-0.85*LL3)/0.009

c_reconstru_water=[PLL3_prime, (PHL3, PLH3, PHH3),(PHL2, PLH2, PHH2),(PLL1, PLH1, PHH1)]

arr, slices = pywt.coeffs_to_array(c_reconstru_water)

img_reconstru_water=pywt.waverec2(c_reconstru_water,'db2',mode='periodization')


#------ ici on essaie de retrouver le watermark avec un affichage protected----------------------

img_with_title= img_reconstru.copy()

img_with_title=putText(img_with_title,"protected")


c_img_with_title=pywt.wavedec2(img_with_title,'db2',mode='periodization',level=3)

RLL3, (RHL3, RLH3, RHH3), (RHL2, RLH2, RHH2), (RLL1, RLH1, RHH1)=c_img_with_title

RLL3_prime=(RLL3-0.85*LL3)/0.009

c_water_reconstru=[RLL3_prime, (RHL3, RLH3, RHH3), (RHL2, RLH2, RHH2), (RLL1, RLH1, RHH1)]

img_water_reconstru=pywt.waverec2(c_water_reconstru,'db2',mode='periodization')



#-------------affichage----------------------


plt.figure()
plt.imshow(arr,cmap=plt.cm.gray)
plt.show()







