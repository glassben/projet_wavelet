import three_level_wavelet_watermark 
import cv2
import numpy as np

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
    
    for i in range(n):
        for j in range(m):
            s+=(image_reconstru[i][j]-image_original[i][j])**2
    
    return (1/(n*m))*s

def PSNR(image_reconstru,image_original):
    return 10*np.log10((255)**2 / MSE(image_reconstru,image_original))
