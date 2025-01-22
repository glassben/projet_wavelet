import cv2
import matplotlib.pyplot as plt

import numpy as np

def embedding_least_significative(file_photo="./image/desktop-wallpaper-full-nature-04-jpeg-1600×900-paysage-coucher-de-soleil-fond-ecran-paysage-nature-paysage.jpg",water_mark="./watermark/fond-ecran-windows-seven-wallpaper-32.jpeg"):

    image =cv2.imread(file_photo,cv2.IMREAD_GRAYSCALE)


    if image.dtype == np.float32:  # if not integer
        image = (image * 255).astype(np.uint8)


    image_water_mark=cv2.imread(water_mark,cv2.IMREAD_GRAYSCALE)

    image_water_mark=cv2.resize(image_water_mark,(image.shape[1],image.shape[0]))

    if image_water_mark.dtype ==np.float32:
        image_water_mark= (image_water_mark*255).astype(np.uint8)


    k=4


    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_water_mark[i,j]=image_water_mark[i,j]>>(8-k)
            image[i,j]=image[i,j]&240 +image_water_mark[i,j]
    
    return image

def desembedding_least_significative(image,k):
        
    image_reconstru=image.copy()

    for i in range(image_reconstru.shape[0]):
        for j in range(image_reconstru.shape[1]):
            image_reconstru[i,j]=image_reconstru[i,j]<<(8-k)

if __name__=="__main__":

    image_reconstru=embedding_least_significative()
    
    cv2.imshow("mon image",image_reconstru)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


