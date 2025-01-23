from three_level_wavelet_watermark import image_embedded_three_level,disembedded_three_level
from Least_Significant_Bit_insertion import embedding_least_significative
from Binary_adapted_watermarking import embedded_Binary_adapted
from combined_DWT_DCT import embedded_combined
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

def putText(image, text):
    (h, w) = image.shape[:2]
    org = (w // 2, h // 2)
    frontScale=2.0
    color=(255.0)
    front=cv2.FONT_HERSHEY_SIMPLEX
    thickness=2
    image = cv2.putText(image, text, org, front, frontScale, color, thickness,cv2.LINE_AA)
    return image



def MSE(image_reconstru,image_original):
    n=image_reconstru.shape[0]
    m=image_reconstru.shape[1]
    s=0
    if image_reconstru.dtype== np.uint8:
        image_reconstru.astype(np.float32)
    if image_original.dtype==np.uint8:
        image_original.astype(np.float32)
    
    for i in range(n):
        for j in range(m):
            s+=(image_reconstru[i][j]-image_original[i][j])**2
    
    return (1/(n*m))*s

def PSNR(image_reconstru,image_original):
    return 10*np.log10((255)**2 / MSE(image_reconstru,image_original))


def correlation_factor(watermarked_extracted,watermarked_original):
    n=watermarked_extracted.shape[0]
    m=watermarked_extracted.shape[1]
    num=0
    denom=0
    for i in range(n):
        for j in range(m):
            num+=watermarked_original[i,j]*watermarked_extracted[i,j]
            denom+=(watermarked_original[i,j])**2
    return num/denom

if __name__=="__main__":
    path_image="./image/target_image1.png"
    path_watermark="./watermark/watermark2.jpg"
    
    watermark_reference=cv2.imread(path_watermark,cv2.IMREAD_GRAYSCALE)
    
    image_reconstruite_three_level,image_real_three_level,real_shape_three_level,alpha=image_embedded_three_level(path_image,path_watermark)
    watermarked_three_level=disembedded_three_level(image_reconstruite_three_level,image_real_three_level,real_shape_three_level)
    print("MSE (three level) is equal to",MSE(watermarked_three_level,watermark_reference))
    print("PSNR (three level) is equal to",PSNR(watermarked_three_level,watermark_reference))
    print("correlation factor is equal to",correlation_factor(watermarked_three_level,watermark_reference))
    print()
    
    image_reconstruite_LSBI,image_real_LSBI,_,_=embedding_least_significative(path_image,path_watermark)
    print("MSE (LSBI) is equal to",MSE(image_reconstruite_LSBI,image_real_LSBI))
    print("PSNR (LSBI) is equal to", PSNR(image_reconstruite_LSBI,image_real_LSBI))
    
    print()
    
    image_reconstruite_combined,image_real_combined,_=embedded_combined(path_image,path_watermark)
    print("MSE (embedded_combined) is equal to",MSE(image_reconstruite_combined,image_real_combined))
    print("PSNR (embedded_combined)is equal to", PSNR(image_reconstruite_combined,image_real_combined))
    
    print()
    
    image_reconstruite_Binary_adapted,image_real_Binary_adapted,_=embedded_Binary_adapted(path_image,path_watermark)
    print("MSE (Binary_adapted) is equal to",MSE(image_reconstruite_Binary_adapted,image_real_Binary_adapted))
    print("PSNR (Binary_adapted)is equal to", PSNR(image_reconstruite_Binary_adapted,image_real_Binary_adapted))
    
    print()
    
    plt.figure()
    plt.imshow(watermarked_three_level,cmap=plt.cm.gray)
    plt.show()
    