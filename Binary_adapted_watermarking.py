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

    
    image_copy=image.copy()
    
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
            if image_water_mark[i,j]>np.std(image_water_mark)*np.sqrt(np.log(image_water_mark.shape[0]*2)):
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

    #print("indice phase de construction")
    #print("indice HL3: ",indices_HL3)
    #print()
    #print("indice_LH3 :",indices_LH3)
    #print()
    pos_HL3=[]
    pos_LH3=[]

    for i,k in enumerate(indices_HL3):
        HL3[k//largeur_hl3][k%largeur_hl3]=HL3[k//largeur_hl3][k%largeur_hl3]+alpha*image_water_mark1[i]
        if image_water_mark1[i]==1:
            pos_HL3.append(k)
    for i,k in enumerate(indices_LH3):
        LH3[k//largeur_lh3][k%largeur_lh3]=LH3[k//largeur_lh3][k%largeur_lh3]+alpha*image_water_mark2[i]
        if image_water_mark2[i]==1:
            pos_LH3.append(k)
        
    
    #print("indice_positive HL3:",pos_HL3)
    
    c_reconstru=[LL4, (HL4,LH4,HH4), (HL3, LH3, HH3), (HL2, LH2, HH2), (LL1, LH1, HH1)]


    img_a_reconstruire=pywt.waverec2(c_reconstru,'haar',mode='periodization')

    return img_a_reconstruire,image_copy,real_shape


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
    HL4_prime=HL4.copy()


    for i in range(hauteur_hl3):
        for j in range(largeur_hl3):
            HL3_prime[i,j]=(1/256)*(HL4_prime[(i//2),(j//2)])

    for i in range(hauteur_lh3):
        for j in range(largeur_lh3):
            LH3_prime[i,j]=(1/256)*(HL4_prime[(i//2),(j//2)])


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


    #print("indice reconstruction")
    #print("indice HL3:" ,indices_HL3)
    #print()
    #print("indice LH3:", indices_LH3)
    #print()
    
    for i,k in enumerate(indices_HL3):
        num=RHL3[k//largeur_hl3][k%largeur_hl3]-HL3[k//largeur_hl3][k%largeur_hl3]
        if num<=0:
            w_reconstru[i]=0
        else:
            w_reconstru[i]=255

    for i,k in enumerate(indices_LH3):
        num=RLH3[k//largeur_lh3][k%largeur_lh3]-LH3[k//largeur_lh3][k%largeur_lh3]
        if num <=0:
            w_reconstru[N//2+i]=0
        else:
            w_reconstru[N//2+i]=255

    w_reconstru=w_reconstru.reshape((largeur_hl3,hauteur_hl3))
    
    
    
    return w_reconstru

#if __name__=="__main__":

    #img_reconstruite,image_original,shape_a_passer=embedded_Binary_adapted()
    
    #watermark_reconstruit=desembedding(img_reconstruite,image_original,shape_a_passer)

    #plt.figure()
    #plt.imshow(watermark_reconstruit,cmap=plt.cm.gray)
    #plt.show()





