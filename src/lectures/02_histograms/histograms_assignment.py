import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def range_by_quantiles(img, p_low, p_high):
    sorted_img = np.sort(img.flatten())

    num_pixels = len(sorted_img)
    index_low = int(num_pixels * p_low)
    index_high = int(num_pixels * p_high)

    x_low = sorted_img[index_low]
    x_high = sorted_img[index_high]

    return x_low, x_high


def transform_by_lut(img, x_low, x_high):
    img_copy = img.copy()
    img_copy[img_copy < x_low] = x_low
    img_copy[img_copy > x_high] = x_high
    img_copy = (img_copy - x_low) / (x_high - x_low)
    return img_copy


'''img = cv.imread(os.path.join("data", "P.jpg"))
grayscaled = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(float) / 255.
P_LOW = 0.01
P_HIGH = 0.99
x_low, x_high = range_by_quantiles(grayscaled, P_LOW, P_HIGH)
transformed = transform_by_lut(grayscaled, x_low, x_high)

plt.figure(figsize=(10,10), dpi=80)
plt.subplot(221)
plt.imshow(grayscaled, cmap="gray")
plt.title(("original '/data/P.jpg'"))
plt.show()'''