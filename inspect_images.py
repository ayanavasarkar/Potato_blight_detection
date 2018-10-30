"""@author ayanava"""

import numpy as np
import matplotlib
import cv2
import matplotlib.pyplot as plt
import os, os.path, imp
import glob

from sklearn.svm import SVC

class img_processing:
    def __init__(self):
        pass

    def show_img(self, img):
        plt.axis("off")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        plt.show()

    def get_images(self, abs_path):
        self.abs_path = abs_path
        if os.path.exists(abs_path):

            self.early_blight_files = os.listdir(self.abs_path)
            print((self.early_blight_files[0]), type(self.early_blight_files))

        else:
            print("Nothing")

    def pre_process(self, w, h):

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.hog=cv2.HOGDescriptor()
        self.hog_feat = np.array([])
        wd = 1000
        ht = 1000
        for i in range(0,len(self.early_blight_files)):

            path = str(self.abs_path) + str(self.early_blight_files[i])

            img = cv2.imread(path, 0)
            #img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            #blur = cv2.GaussianBlur(img, (5, 5), 0)
            #cl = self.clahe.apply(blur)
            #bilateral = cv2.bilateralFilter(cl, 9, 75, 75)

            # h = self.hog.compute(img, winStride=(16, 16), padding=(0, 0))
            # h_trans = h.transpose()
            # self.hog_feat = np.vstack(h_trans)

        self.show_img(img)
    def check_ht_wd(self):

        wd = 1000
        ht = 1000
        for i in range(0, len(self.early_blight_files)):
            path = str(self.abs_path) + str(self.early_blight_files[i])

            img = cv2.imread(path, 0)

            height, width = img.shape[:2]
            if (height <= ht):
                ht = height
            if (width <= wd):
                wd = width

            height = 0
            width = 0
        self.show_img(img)
        return (ht, wd)

obj = img_processing()

early_blight_path = "Potato___Early_blight/"
healthy_path = "Potato___healthy/"
late_blight_path = "Potato___Late_blight/"


obj.get_images(healthy_path)

w = 256
h = 256
ht, wd = obj.check_ht_wd()
# print(hog_feat.shape)
print(ht, wd)
exit(0)

#back_sub2 = cv2.createBackgroundSubtractorMOG2()

img1 = cv2.imread(early_blight_path,0)
img2 = cv2.imread(healthy_path)
img3 = cv2.imread(late_blight_path)

#img1 = back_sub2.apply(img1)

obj.show_img(img1)
#back = back_sub2.apply(img1)

# plt.axis("off")
# plt.imshow(back, cmap='binary')
# plt.show()

blur1 = cv2.GaussianBlur(img1,(5,5),0)
obj.show_img(blur1)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(blur1)
obj.show_img(cl)

bilateral = cv2.bilateralFilter(cl,9,75,75)
obj.show_img(bilateral)



#######To be tried
blur1 = cv2.GaussianBlur(img1,(2,2),0)
median = cv2.medianBlur(img1,5)
blur2 = cv2.bilateralFilter(img1,25,100,100)

canny = cv2.Canny(img1, 100, 200)

plt.imshow(cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB))
plt.show()

ret, thresh = cv2.threshold(img1,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
th2 = cv2.adaptiveThreshold(thresh,1,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img1,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

exit(0)
# loop over the threshold methods
for (threshName, threshMethod) in methods:
    # threshold the image and show it
    (T, thresh) = cv2.threshold(img1, 245, 255, threshMethod)

    plt.imshow(thresh)
    plt.show()