import cv2
import matplotlib.pyplot as plt
import pywt
import numpy as np

#-------------début chargement photo--------------------------------------------

def embedded_Binary_adapted(file_photo="./image/desktop-wallpaper-full-nature-04-jpeg-1600×900-paysage-coucher-de-soleil-fond-ecran-paysage-nature-paysage.jpg",water_mark="./watermark/A-sample-binary-watermark-logo-image.png",alpha=30):

    image =cv2.imread(file_photo,cv2.IMREAD_GRAYSCALE)

    if image.dtype == np.float32:  # if not integer
        image = (image * 255).astype(np.uint8)


    image=cv2.resize(image,(256,256))

    c = pywt.wavedec2(image, 'haar', mode='periodization',level=4)

    LL4, (HL4,LH4,HH4), (HL3, LH3, HH3), (HL2, LH2, HH2), (LL1, LH1, HH1) =c







    image_water_mark=cv2.imread(water_mark,cv2.IMREAD_GRAYSCALE)

    real_shape=image_water_mark.shape

    longueur_reduced_water_shape,largeur_reduced_water_shape=HL3.shape

    image_water_mark=cv2.resize(image_water_mark,(largeur_reduced_water_shape,longueur_reduced_water_shape))



#----------------- watermarked embedding -----------------------------------

    N=longueur_reduced_water_shape*largeur_reduced_water_shape
    
    w=np.zeros(longueur_reduced_water_shape*largeur_reduced_water_shape)


    for i in range(image_water_mark.shape[0]):
        for j in range(image_water_mark.shape[1]):
            if image_water_mark[i,j]==255:
                w[image_water_mark.shape[1]*i+j]=1
            else:
                w[image_water_mark.shape[1]*i+j]=-1







    image_water_mark1=w[0:N//2+1] #w1 coefficient à mettre dans I2⁰ 
    image_water_mark2=w[N//2+1:] # w2 coefficient à mettre dans I2²


    HL3_prime=HL3.copy()
    LH3_prime=LH3.copy()
    HL4_prime=HL4.copy()



    hauteur_hl3,largeur_hl3=HL3_prime.shape
    hauteur_lh3,largeur_lh3=LH3_prime.shape
    







    for i in range(hauteur_hl3):
        for j in range(largeur_hl3):
            HL3_prime[i,j]=(1/256)*HL4_prime[(i//2),(j//2)]

    for i in range(hauteur_lh3):
        for j in range(largeur_lh3):
            LH3_prime[i,j]=(1/256)*HL4_prime[(i//2),(j//2)]


    HL3_prime=HL3_prime.flatten()
    LH3_prime=LH3_prime.flatten()

    indices_HL3=np.zeros(len(image_water_mark1),dtype=int)
    indices_LH3=np.zeros(len(image_water_mark2),dtype=int)


    for n in range(len(image_water_mark1)):
        my_min=int(np.nanargmin(HL3_prime))
        indices_HL3[n]=my_min
        HL3_prime[my_min]=np.nan

    for n in range(len(image_water_mark2)):
        my_min=int(np.nanargmin(LH3_prime))
        indices_LH3[n]=my_min
        LH3_prime[my_min]=np.nan




    for i,k in enumerate(indices_HL3):
        HL3[k//largeur_hl3][k%largeur_hl3]=HL3[k//largeur_hl3][k%largeur_hl3]+alpha*image_water_mark1[i]

    for i,k in enumerate(indices_LH3):
        LH3[k//largeur_lh3][k%largeur_lh3]=LH3[k//largeur_lh3][k%largeur_lh3]+alpha*image_water_mark2[i]


    c_reconstru=[LL4, (HL4,LH4,HH4), (HL3, LH3, HH3), (HL2, LH2, HH2), (LL1, LH1, HH1)]


    img_a_reconstruire=pywt.waverec2(c_reconstru,'haar',mode='periodization')

    return img_a_reconstruire,image,real_shape


#-------------------- watermarked desembedding ------------------------------#

def desembedding(img_reconstru,image,shape_watermark):


    c_a_reconstruire = pywt.wavedec2(img_reconstru, 'haar', mode='periodization',level=4)


    [RLL4, (RHL4,RLH4,RHH4), (RHL3, RLH3, RHH3), (RHL2, RLH2, RHH2), (RLL1, RLH1, RHH1)]= c_a_reconstruire


    c = pywt.wavedec2(image, 'haar', mode='periodization',level=4)

    LL4, (HL4,LH4,HH4), (HL3, LH3, HH3), (HL2, LH2, HH2), (LL1, LH1, HH1) = c

    hauteur_hl3,largeur_hl3=RHL3.shape
    hauteur_lh3,largeur_lh3=RLH3.shape
    N=hauteur_hl3*largeur_hl3
    w_reconstru=np.zeros(N)




    HL3_prime=HL3.copy()
    LH3_prime=LH3.copy()



    for i in range(hauteur_hl3):
        for j in range(largeur_hl3):
            HL3_prime[i,j]=(1/256)*(HL4[(i//2),(j//2)])

    for i in range(hauteur_lh3):
        for j in range(largeur_lh3):
            LH3_prime[i,j]=(1/256)*(HL4[(i//2),(j//2)])


    HL3_prime=HL3_prime.flatten()
    LH3_prime=LH3_prime.flatten()

    indices_HL3=np.zeros(N//2,dtype=int)
    indices_LH3=np.zeros(N//2,dtype=int)

    for n in range(N//2):
        my_min=int(np.nanargmin(HL3_prime))
        indices_HL3[n]=my_min
        HL3_prime[my_min]=np.nan

    for n in range(N//2):
        my_min=int(np.nanargmin(LH3_prime))
        indices_LH3[n]=my_min
        LH3_prime[my_min]=np.nan



    for i,k in enumerate(indices_HL3):
        num=RHL3[k//largeur_hl3][k%largeur_hl3]-HL3[k//largeur_hl3][k%largeur_hl3]
        if num<=0:
            w_reconstru[i]=0
        else:
            w_reconstru[i]=1

    for i,k in enumerate(indices_LH3):
        num=LH3[k//largeur_lh3][k%largeur_lh3]-LH3[k//largeur_lh3][k%largeur_lh3]
        if num <=0:
            w_reconstru[N//2+i]=0
        else:
            w_reconstru[N//2+i]=1

    w_reconstru=w_reconstru.reshape((largeur_hl3,hauteur_hl3))
    
    return w_reconstru

#if __name__=="__main__":

    #img_reconstruite,image_original,shape_a_passer=embedded_Binary_adapted()
    
    #watermark_reconstruit=desembedding(img_reconstruite,image_original,shape_a_passer)

    #plt.figure()
    #plt.imshow(watermark_reconstruit,cmap=plt.cm.gray)
    #plt.show()





