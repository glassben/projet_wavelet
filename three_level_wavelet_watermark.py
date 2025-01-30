import numpy as np
from numpy.linalg import norm 
import matplotlib.pyplot as plt
import pywt
import cv2
from scipy import ndimage




def image_embedded_three_level(file_photo="./image/target_image2.jpg"
,water_mark="./watermark/watermark2.jpg",alpha=0.85,beta=0.009):


    image =cv2.imread(file_photo,cv2.IMREAD_GRAYSCALE)
    
    #image=cv2.resize(image,(256,256))

    #image_rotated=ndimage.rotate(image,45)

    image_water_mark=cv2.imread(water_mark,cv2.IMREAD_GRAYSCALE)
    
    
    real_shape=image_water_mark.shape

    image_water_mark=cv2.resize(image_water_mark,(image.shape[1],image.shape[0]))





    c = pywt.wavedec2(image, 'haar', mode='periodization',level=3)

    c_water_mark=pywt.wavedec2(image_water_mark, 'haar', mode='periodization',level=3)




    LL3, (HL3, LH3, HH3), (HL2, LH2, HH2), (LL1, LH1, HH1) = c

    WLL3, (WHL3, WLH3, WHH3), (WHL2, WLH2, WHH2), (WLL1, WLH1, WHH1)=c_water_mark

#----------- ici on essaie de camufler le watermark ------------------

    LL3_prime=alpha*LL3+beta*WLL3

    c_reconstru=[LL3_prime, (HL3, LH3, HH3), (HL2, LH2, HH2), (LL1, LH1, HH1)]


    img_reconstru=pywt.waverec2(c_reconstru,'haar',mode='periodization')

    return img_reconstru,image,real_shape,alpha

#---ici on essaie de retrouver le watermark---------------

def disembedded_three_level(img_a_reconstruire,image_original,shape_watermark,alpha_coeff=0.85):

    img_reconstru_bis=img_a_reconstruire.copy()
    
    c = pywt.wavedec2(image_original, 'haar', mode='periodization',level=3)
    
    LL3, (HL3, LH3, HH3), (HL2, LH2, HH2), (LL1, LH1, HH1) = c
    
    c_img_reconstru_bis=pywt.wavedec2(img_reconstru_bis,'haar',mode='periodization',level=3)

    PLL3, (PHL3, PLH3, PHH3), (PHL2, PLH2, PHH2), (PLL1, PLH1, PHH1)=c_img_reconstru_bis



    PLL3_prime=(PLL3-alpha_coeff*LL3)

    #c_reconstru_water=[PLL3_prime, (PHL3, PLH3, PHH3),(PHL2, PLH2, PHH2),(PLL1, PLH1, PHH1)]
    #PLL3_prime=cv2.resize(PLL3_prime,(shape_watermark[1],shape_watermark[0]))
    

    #img_reconstru_water=pywt.waverec2(c_reconstru_water,'db2',mode='periodization')
    PLL3_prime=cv2.resize(PLL3_prime,(shape_watermark[1],shape_watermark[0]))
    
    return PLL3_prime






#-------------affichage----------------------

#if __name__=="__main__":

    #img_reconstru,image_originale,real_shape,alpha_reconstru=image_embedded_three_level()
    #watermark_reconstru=disembedded_three_level(img_reconstru,image_originale,real_shape,alpha_reconstru)
    
    #plt.figure()
    #plt.imshow(watermark_reconstru,cmap=plt.cm.gray)
    #plt.show()