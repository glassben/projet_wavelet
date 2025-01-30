from three_level_wavelet_watermark import image_embedded_three_level,disembedded_three_level
from Least_Significant_Bit_insertion import embedding_least_significative,desembedding_least_significative
from Binary_adapted_watermarking import embedded_Binary_adapted,desembedding
from combined_DWT_DCT import embedded_combined,disembedded
import cv2
import numpy as np
import matplotlib.pyplot as plt

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def noisy(image,sigma): # add a normal noise with standard variation sigma
    return image + np.random.normal(0, sigma, size=image.shape)


def crop(image,factor): # here we crop the image by selecting the value which are contained in a window [h//2-h//(2**factor),h//2+h//(2**factor)]*[l//2-l//(2**factor),l//2+l//(2**factor)]
    h,l=image.shape
    image=image[h//2-h//(2**factor):h//2+h//(2**factor),l//2-l//(2**factor):l//2+l//(2**factor)]
    image=cv2.resize(image,(l,h)) # We resize the image 
    return image



def MSE(image_reconstru,image_original):
    image_copy=image_original.copy()
    
    n=image_reconstru.shape[0]
    m=image_reconstru.shape[1]
    
    
    if image_original.shape!=image_reconstru.shape:
        image_original=image_copy.resize((m,n))
    
    s=0
    if image_reconstru.dtype== np.uint8:
        image_reconstru.astype(np.float32)
    if image_copy.dtype==np.uint8:
        image_copy.astype(np.float32)
    
    for i in range(n):
        for j in range(m):
            s+=(float(image_reconstru[i][j])-float(image_copy[i][j]))**2
    return (1/(n*m))*s

def PSNR(image_reconstru,image_original):
    return 10*np.log10((255)**2 / MSE(image_reconstru,image_original))


def correlation_factor(watermarked_extracted,watermarked_original):
    n=watermarked_extracted.shape[0]
    m=watermarked_extracted.shape[1]
    watermarked_copy=watermarked_original.copy()
    
    if watermarked_extracted.shape!=watermarked_original.shape:
        watermarked_original=cv2.resize(watermarked_copy,(m,n))
    
    num=0
    denom1=0
    denom2=0
    
    for i in range(n):
        for j in range(m):
            num+=float(watermarked_extracted[i,j])*float(watermarked_copy[i,j])
            denom2+=(watermarked_copy[i,j])**2
            denom1+=(watermarked_extracted[i,j])**2
    
    return num/(np.sqrt(denom2)*np.sqrt(denom1))

if __name__=="__main__":
    path_image="./image/target_image1.png"
    path_watermark="./watermark/watermark1.jpg"
    
    #sigma=20.0
    #factor=4
    #theta=0.5
    
    
    watermark_reference=cv2.imread(path_watermark,cv2.IMREAD_GRAYSCALE)
    
    image_reconstruite_three_level,image_real_three_level,real_shape_three_level,alpha=image_embedded_three_level(path_image,path_watermark,0.99)
    #image_reconstruite_three_level=noisy(image_reconstruite_three_level,sigma)
    #image_reconstruite_three_level=crop(image_reconstruite_three_level,factor)
    #image_reconstruite_three_level=rotate(image_reconstruite_three_level,theta)
    watermarked_three_level=disembedded_three_level(image_reconstruite_three_level,image_real_three_level,real_shape_three_level,alpha)
    #print("MSE (three level) is equal to",MSE(watermarked_three_level,watermark_reference))
    #print("PSNR (three level) is equal to",PSNR(watermarked_three_level,watermark_reference))
    print("correlation factor is equal to",correlation_factor(watermarked_three_level,watermark_reference))
    print("MSE (three level) of watermarked", MSE(watermarked_three_level,watermark_reference))
    print("PNSR (three level) of watermarked", PSNR(watermarked_three_level,watermark_reference))
    
    
    print()
    
    image_reconstruite_LSBI,image_real_LSBI,real_shape_LSBI,k_LSBI=embedding_least_significative(path_image,path_watermark)
    #image_reconstruite_LSBI=noisy(image_reconstruite_LSBI,sigma)
    #image_reconstruite_LSBI=crop(image_reconstruite_LSBI,factor)
    #image_reconstruite_LSBI=rotate(image_reconstruite_LSBI,theta)
    watermarked_LSBI=desembedding_least_significative(image_reconstruite_LSBI,k_LSBI,real_shape_LSBI)
    #print("MSE (LSBIwatermarked_LSBI) is equal to",MSE(image_reconstruite_LSBI,image_real_LSBI))
    #print("PSNR (LSBI) is equal to", PSNR(image_reconstruite_LSBI,image_real_LSBI))
    print("correlation factor is equal to",correlation_factor(watermarked_LSBI,watermark_reference))
    print("MSE (LSBI) of watermarked is equal to",MSE(watermarked_LSBI,watermark_reference))
    print("PSNR (LSBI) of watermarked is equal to", PSNR(watermarked_LSBI,watermark_reference))
    
    
    print()
    
    image_reconstruite_combined,image_real_combined,real_shape_combined=embedded_combined(path_image,path_watermark)
    #image_reconstruite_combined=noisy(image_reconstruite_combined,sigma)
    #image_reconstruite_combined=crop(image_reconstruite_combined,factor)
    #image_reconstruite_combined=rotate(image_reconstruite_combined,theta)
    watermarked_combined=disembedded(image_reconstruite_combined,real_shape_combined)
    #print("MSE (embedded_combined) is equal to",MSE(image_reconstruite_combined,image_real_combined))
    #print("PSNR (embedded_combined)is equal to", PSNR(image_reconstruite_combined,image_real_combined))
    print("correlation factor is equal to",correlation_factor(watermarked_combined,watermark_reference))
    print("MSE (embedded_combined) of watermarked is equal to", MSE(watermarked_combined,watermark_reference))
    print("PSNR (embedded_combined) of watermarked is equal to", PSNR(watermarked_combined,watermark_reference))
    
    print()
    
    image_reconstruite_Binary_adapted,image_real_Binary_adapted,real_shape_Binary_adapted=embedded_Binary_adapted(path_image,path_watermark)
    #image_real_Binary_adapted=noisy(image_reconstruite_Binary_adapted,sigma)
    #image_real_Binary_adapted=crop(image_reconstruite_Binary_adapted,factor)
    #image_real_Binary_adapted=rotate(image_real_Binary_adapted,theta)
    watermarked_Binary_adapted=desembedding(image_reconstruite_Binary_adapted,image_real_Binary_adapted,real_shape_Binary_adapted)
    #print("MSE (Binary_adapted) is equal to",MSE(image_reconstruite_Binary_adapted,image_real_Binary_adapted))
    #print("PSNR (Binary_adapted)is equal to", PSNR(image_reconstruite_Binary_adapted,image_real_Binary_adapted))
    print("correlation factor is equal to",correlation_factor(watermarked_Binary_adapted,watermark_reference))
    print("MSE (Binary_adapted) of watermarked is equal to", MSE(watermarked_Binary_adapted,watermark_reference))
    print("PSNR (Binary_adapted) of watermarked is equal to", PSNR(watermarked_Binary_adapted,watermark_reference))

    
    print()
    
    plt.figure()
    plt.imshow(watermarked_three_level,cmap=plt.cm.gray)
    plt.show()
    