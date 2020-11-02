# a sample file to see how the scanner module works

import cv2
import os
import scanner as scan

img = cv2.imread('doc11.jpg')
images, names = scan.get_scanned_image(img)
scan.show_images(images, names, True)
scanned_image = images[1]
os.chdir('c:/Users/hp-2111/Desktop/output3/')
new1 = scan.change_brightness_contrast(scanned_image, 0, 1.25)
new2 = scan.change_brightness_contrast(scanned_image, 50, 1)
cv2.imwrite('new1.jpg', new1)
cv2.imwrite('new2.jpg', new2)
