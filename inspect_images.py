import numpy as np
import matplotlib
import cv2
import matplotlib.pyplot as plt
import os, os.path
import glob


class img_processing:
    def __init__(self):
        pass

    def get_images():

        if os.path.exists(abs_path):

            early_blight_files = os.listdir(abs_path)
            print(len(early_blight_files))

        else:
            print("Nothing")



early_blight_path = "Potato___Early_blight/2.jpg"
healthy_path = "Potato___healthy/1.jpg"
late_blight_path = "Potato__Late_blight/1.jpg"

#back_sub2 = cv2.createBackgroundSubtractorMOG2()


img1 = cv2.imread(early_blight_path,0)
img2 = cv2.imread(healthy_path)
img3 = cv2.imread(late_blight_path)

#img1 = back_sub2.apply(img1)

# plt.axis("off")
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR))
plt.show()

#back = back_sub2.apply(img1)

# plt.axis("off")
# plt.imshow(back, cmap='binary')
# plt.show()

blur1 = cv2.GaussianBlur(img1,(5,5),0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
cl = clahe.apply(blur1)
plt.imshow(cv2.cvtColor(cl, cv2.COLOR_GRAY2RGB))
plt.show()
exit(0)

blur1 = cv2.GaussianBlur(img1,(5,5),0)
median = cv2.medianBlur(img1,5)
blur2 = cv2.bilateralFilter(img1,25,100,100)

plt.imshow(cv2.cvtColor(blur2, cv2.COLOR_BGR2RGB))
plt.show()

plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
plt.show()
exit(0)

canny = cv2.Canny(img1, 100, 200)

plt.imshow(cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB))
plt.show()

ret, thresh = cv2.threshold(img1,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
th2 = cv2.adaptiveThreshold(thresh,1,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img1,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

plt.imshow(th2)
plt.show()

plt.imshow(th3)
plt.show()
exit(0)
# loop over the threshold methods
for (threshName, threshMethod) in methods:
    # threshold the image and show it
    (T, thresh) = cv2.threshold(img1, 245, 255, threshMethod)

    plt.imshow(thresh)
    plt.show()