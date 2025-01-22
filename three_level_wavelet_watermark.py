import numpy as np
from numpy.linalg import norm 
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


def image_embedded_three_level(file_photo="./image/desktop-wallpaper-full-nature-04-jpeg-1600×900-paysage-coucher-de-soleil-fond-ecran-paysage-nature-paysage.jpg"
,water_mark="./watermark/fond-ecran-windows-seven-wallpaper-32.jpeg"):


    image =cv2.imread(file_photo,cv2.IMREAD_GRAYSCALE)

    #image_rotated=ndimage.rotate(image,45)

    image_water_mark=cv2.imread(water_mark,cv2.IMREAD_GRAYSCALE)

    image_water_mark=cv2.resize(image_water_mark,(image.shape[1],image.shape[0]))





    c = pywt.wavedec2(image, 'db2', mode='periodization',level=3)

    c_water_mark=pywt.wavedec2(image_water_mark, 'db2', mode='periodization',level=3)




    LL3, (HL3, LH3, HH3), (HL2, LH2, HH2), (LL1, LH1, HH1) = c

    WLL3, (WHL3, WLH3, WHH3), (WHL2, WLH2, WHH2), (WLL1, WLH1, WHH1)=c_water_mark

#----------- ici on essaie de camufler le watermark ------------------

    LL3_prime=0.85*LL3+0.009*WLL3

    c_reconstru=[LL3_prime, (HL3, LH3, HH3), (HL2, LH2, HH2), (LL1, LH1, HH1)]


    img_reconstru=pywt.waverec2(c_reconstru,'db2',mode='periodization')

    return img_reconstru,image

#---ici on essaie de retrouver le watermark---------------

def disembedded_three_level(img_reconstru,image):

    img_reconstru_bis=img_reconstru.copy()
    
    c = pywt.wavedec2(image, 'db2', mode='periodization',level=3)
    
    LL3, (HL3, LH3, HH3), (HL2, LH2, HH2), (LL1, LH1, HH1) = c
    
    c_img_reconstru_bis=pywt.wavedec2(img_reconstru_bis,'db2',mode='periodization',level=3)

    PLL3, (PHL3, PLH3, PHH3), (PHL2, PLH2, PHH2), (PLL1, PLH1, PHH1)=c_img_reconstru_bis



    PLL3_prime=(PLL3-0.85*LL3)/0.009

    c_reconstru_water=[PLL3_prime, (PHL3, PLH3, PHH3),(PHL2, PLH2, PHH2),(PLL1, PLH1, PHH1)]

    

    img_reconstru_water=pywt.waverec2(c_reconstru_water,'db2',mode='periodization')

    return img_reconstru



#-----------calcul PSNR and MSE ------------------------------


def MSE(image_reconstru,image_original):
    n=image_reconstru.shape[0]
    m=image_reconstru.shape[1]
    s=0
    
    for i in range(n):
        for j in range(m):
            s+=(image_reconstru[i][j]-image_original[i][j])**2
    
    return (1/(n*m))*s

def PSNR(image_reconstru,image_original):
    return 10*np.log10((255)**2 / MSE(image_reconstru,image_original))


#-------------affichage----------------------

if __name__=="__main__":

    img_reconstru,image=image_embedded_three_level()
    plt.figure()
    plt.imshow(img_reconstru,cmap=plt.cm.gray)
    print("MSE est de ",MSE(img_reconstru,image))
    print("PSNR est de ",PSNR(img_reconstru,image))
    plt.show()







