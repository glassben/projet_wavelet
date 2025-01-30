import cv2
import matplotlib.pyplot as plt
import pywt
import numpy as np
from scipy.fftpack import dct,idct
from random import seed,randint,random


def embedded_combined(file_photo="./image/desktop-wallpaper-full-nature-04-jpeg-1600×900-paysage-coucher-de-soleil-fond-ecran-paysage-nature-paysage.jpg"
,water_mark="./watermark/A-sample-binary-watermark-logo-image.png",k=30):



    image =cv2.imread(file_photo,cv2.IMREAD_GRAYSCALE)

    if image.dtype == np.float32:  # if not integer
        image = (image * 255).astype(np.uint8)

    image=cv2.resize(image,(256,256))
    # decomposition en 4

    c = pywt.wavedec2(image, 'haar', mode='periodization',level=1)

    LL1,(HL1,LH1,HH1)=c

    # On décompose la partie supérieure droite HL1

    cHL=pywt.wavedec2(HL1, 'haar', mode='periodization',level=1)

    LL,(HL,LH,HH)=cHL

    # on va imbriqué le watermark dans la partie supérieure droite HL



    image_water_mark=cv2.imread(water_mark,cv2.IMREAD_GRAYSCALE)

    real_shape=image_water_mark.shape

    
    longueur_reduced_water_shape,largeur_reduced_water_shape=HL.shape

# on adapte la taille de l'image watermark à la taille de HL 
    image_water_mark=cv2.resize(image_water_mark,(largeur_reduced_water_shape,longueur_reduced_water_shape))


            


    image_water_mark=image_water_mark.flatten()


    longueur_HL,largeur_HL=HL.shape



    # calcul de la DCT de HL

    all_subdct=np.empty((longueur_HL,largeur_HL))



    for i in range(0,longueur_HL,4):
        for j in range(0,largeur_HL,4):
            subpixels=HL[i:i+4,j:j+4]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            all_subdct[i:i+4, j:j+4] = subdct



    

    maxi=0
    for i in range(0,longueur_HL,4):
        for j in range(0,largeur_HL,4):
            subpixels=all_subdct[i:i+4, j:j+4]
            maxi=max(subpixels[1,2],maxi)
            maxi=max(subpixels[1,3],maxi)
            maxi=max(subpixels[2,0],maxi)
            maxi=max(subpixels[2,1],maxi)

    seed(4)
    one=[random(),random(),random(),random()]
    zero=[random(),random(),random(),random()]

    #print(one)
    #print(zero)

    ind=0
    for x in range(0,longueur_HL,4):
        for y in range(0,largeur_HL,4):
            if ind < len(image_water_mark):
                subdct=all_subdct[x:x+4,y:y+4]
                if image_water_mark[ind]==255:
                    subdct[1,2]=subdct[1,2]+k*one[0]
                    subdct[1,3]=subdct[1,3]+k*one[1]
                    subdct[2,0]=subdct[2,0]+k*one[2]
                    subdct[2,1]=subdct[2,1]+k*one[3]
                else :
                    subdct[1,2]=subdct[1,2]+k*zero[0]
                    subdct[1,3]=subdct[1,3]+k*zero[1]
                    subdct[2,0]=subdct[2,0]+k*zero[2]
                    subdct[2,1]=subdct[2,1]+k*zero[3]
                
                all_subdct[x:x+4, y:y+4] = subdct
                ind += 1




    for x in range(0,longueur_HL,4):
        for y in range(0,largeur_HL,4):
            subidct=idct(idct(all_subdct[x:x+4, y:y+4].T, norm="ortho").T, norm="ortho")
            all_subdct[x:x+4, y:y+4]=subidct



    cHL=[LL,(all_subdct,LH,HH)]

    HL1=pywt.waverec2(cHL, 'haar', mode='periodization')

    c=[LL1,(HL1,LH1,HH1)]

    image_reconstru=pywt.waverec2(c, 'haar', mode='periodization')

    return image_reconstru,image,real_shape


def disembedded(image_embedded,real_shape):
    c_reconstru = pywt.wavedec2(image_embedded, 'haar', mode='periodization',level=1)
    
    RLL1,(RHL1,RLH1,RHH1)=c_reconstru
    
    RcHL=pywt.wavedec2(RHL1, 'haar', mode='periodization',level=1)

    RLL,(RHL,RLH,RHH)=RcHL
    
    longueur_RHL,largeur_RHL=RHL.shape
    
    watermarked=np.zeros(longueur_RHL*largeur_RHL)
    
    all_subdct=[]
    
    
    maxi=0
    for i in range(0,longueur_RHL,4):
        for j in range(0,largeur_RHL,4):
            subpixels=RHL[i:i+4,j:j+4]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            coeff=[subdct[1,2],subdct[1,3],subdct[2,0],subdct[2,1]]
            all_subdct.append(coeff)
            maxi=max(subdct[1,2],maxi)
            maxi=max(subdct[1,3],maxi)
            maxi=max(subdct[2,0],maxi)
            maxi=max(subdct[2,1],maxi)
            
    
    seed(4)
    one=[random(),random(),random(),random()]
    zero=[random(),random(),random(),random()]
    
    
    #print(one)
    #print(zero)
    
    for i,x in enumerate(all_subdct):
        corr1=np.corrcoef(x,one)[0,1]
        corr0=np.corrcoef(x,zero)[0,1]
        if corr1>corr0:
            watermarked[i]=255
        else:
            watermarked[i]=0
    
    watermarked=cv2.resize(watermarked,(real_shape[1],real_shape[0]))
    
    return watermarked
            
   
    





#if __name__=="__main__":

    #image_embedded,shape_return=embedded_combined()
    #watermark_reconstruit=disembedded(image_embedded,shape_return)
    
    
    
    #plt.figure()
    #plt.imshow(watermark_reconstruit,cmap=plt.cm.gray)
    #plt.show()



#  reconstru 








