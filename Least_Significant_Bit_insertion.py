import cv2


import numpy as np

def embedding_least_significative(file_photo="./image/desktop-wallpaper-full-nature-04-jpeg-1600×900-paysage-coucher-de-soleil-fond-ecran-paysage-nature-paysage.jpg",water_mark="./watermark/A-sample-binary-watermark-logo-image.png",k=4):

    
    image =cv2.imread(file_photo,cv2.IMREAD_GRAYSCALE)


    if image.dtype == np.float32:  # if not integer
        image = (image * 255).astype(np.uint8)

    image.resize((256,256))
    
    image_initial=image.copy()
    
    image_water_mark=cv2.imread(water_mark,cv2.IMREAD_GRAYSCALE)
    
    real_shape=image_water_mark.shape

    image_water_mark=cv2.resize(image_water_mark,(image.shape[1],image.shape[0]))

    if image_water_mark.dtype ==np.float32:
        image_water_mark= (image_water_mark*255).astype(np.uint8)


    


    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            number=0
            for u in range(k):
                number+=2**(u)
            image_water_mark[i,j]=image_water_mark[i,j]>>(8-k)
            image[i,j]=image[i,j]&(255-number) +image_water_mark[i,j]
    
    return image,image_initial,real_shape,k

def desembedding_least_significative(image_a_reconstruire,k,shape_reconstru_water):
        
    image_reconstru=image_a_reconstruire.copy()

    for i in range(image_reconstru.shape[0]):
        for j in range(image_reconstru.shape[1]):
            image_reconstru[i,j]=image_reconstru[i,j]<<(8-k)
    
    image_reconstru=cv2.resize(image_reconstru,(shape_reconstru_water[1],shape_reconstru_water[0]))
    return image_reconstru

#if __name__=="__main__":
    #coeff=4
    #image_a_reconstruit,coeff,shape_water=embedding_least_significative(k=coeff)
    
    #new_watermark=desembedding_least_significative(image_a_reconstruit,coeff,shape_water)
    
    #cv2.imshow("mon image",new_watermark)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


