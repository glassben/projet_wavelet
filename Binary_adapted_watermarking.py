import cv2
import matplotlib.pyplot as plt
import pywt
import numpy as np



file_photo="./image/desktop-wallpaper-full-nature-04-jpeg-1600×900-paysage-coucher-de-soleil-fond-ecran-paysage-nature-paysage.jpg"


water_mark="./watermark/A-sample-binary-watermark-logo-image.png"

image =cv2.imread(file_photo,cv2.IMREAD_GRAYSCALE)


if image.dtype == np.float32:  # if not integer
        image = (image * 255).astype(np.uint8)



c = pywt.wavedec2(image, 'db2', mode='periodization',level=4)

LL4, (HL4,LH4,HH4), (HL3, LH3, HH3), (HL2, LH2, HH2), (LL1, LH1, HH1) =c




image_water_mark=cv2.imread(water_mark,cv2.IMREAD_GRAYSCALE)

image_water_mark=cv2.resize(image_water_mark,(HL3.shape[1],HL3.shape[0]))

w=np.zeros(image_water_mark.shape[0]*image_water_mark.shape[1])

for i in range(image_water_mark.shape[0]):
    for j in range(image_water_mark.shape[1]):
        if image_water_mark[i,j]==255:
            w[image_water_mark.shape[1]*i+j]=1
        else:
            w[image_water_mark.shape[1]*i+j]=-1




alpha=30



N=len(w)

image_water_mark1=w[0:N//2+1]
image_water_mark2=w[N//2+1:]

HL3_prime=HL3.copy()



LH3_prime=LH3.copy()



HL4_prime=HL4.copy()

hauteur_hl3,largeur_hl3=HL3_prime.shape
hauteur_lh3,largeur_lh3=LH3_prime.shape
hauteur_hl4,largeur_hl4=HL4_prime.shape







for i in range(hauteur_hl3):
    for j in range(largeur_hl3):
        HL3_prime[i,j]=(1/256)*HL4_prime[(i//2),(j//2)]

for i in range(hauteur_lh3):
    for j in range(largeur_lh3):
        LH3_prime[i,j]=(1/256)*LH3_prime[(i//2),(j//2)]


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
    HL3[k//largeur_hl3][k%largeur_hl3]=HL3[k//largeur_hl3][k%largeur_hl3]+30*image_water_mark1[i]

for i,k in enumerate(indices_LH3):
    LH3[k//largeur_lh3][k%largeur_lh3]=LH3[k//largeur_lh3][k%largeur_lh3]+30*image_water_mark2[i]


c_reconstru=[LL4, (HL4,LH4,HH4), (HL3, LH3, HH3), (HL2, LH2, HH2), (LL1, LH1, HH1)]


img_reconstru=pywt.waverec2(c_reconstru,'db2',mode='periodization')


plt.figure()
plt.imshow(img_reconstru,cmap=plt.cm.gray)
plt.show()





