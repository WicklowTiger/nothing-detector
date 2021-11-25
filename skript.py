import cv2
import types
import numpy as np
import rawpy
import imageio


def rgb_2_bayer(image: np.ndarray):
    new_image = np.zeros(shape=(image.shape[0]*2, image.shape[1]*2, 3), dtype=np.uint8)
    for h in range(0, image.shape[0]):
        for w in range(0, image.shape[1]):
            new_image[h*2][w*2] = [image[h][w][0], 0, 0]
            new_image[h*2][w*2 + 1] = [0, image[h][w][1]/2, 0]
            new_image[h*2 + 1][w*2] = [0, image[h][w][1]/2, 0]
            new_image[h*2][w*2 + 1] = [0, 0, image[h][w][2]]
    return new_image


path = 'images_bayer/sample.dng'
image = rawpy.imread(path).raw_image.copy()
# rgb = rawpy.imread(path).raw_image.copy()

# image = cv2.resize(image, dsize=(1280, 720),interpolation = cv2.INTER_AREA)


print(image.shape)


image = image / 64

image = image.astype('uint8')
print(image.shape)

# rgb = cv2.cvtColor(rgb, cv2.COLOR_BAYER_RG2RGB)
# rgb = cv2.resize(rgb, dsize=(1280, 720), interpolation=cv2.INTER_AREA)
cv2.imshow('bayer', image)
cv2.imwrite('saved_images/image_00.jpg', image)
# cv2.imshow('rgb', rgb)
cv2.waitKey(0)
#
# print(image.dtype)
# print(image)
