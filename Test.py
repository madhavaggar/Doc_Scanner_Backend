# a sample file to see how the scanner module works

import cv2
import os
import scanner as scan

# type any image name here
img = cv2.imread('doc11.jpg')

# to get the scanned image and other images (its binary output, etc)
scanned_image, binary_output, images, names = scan.get_scanned_image(img)

# you can set any path here to a folder
os.chdir('c:/Users/hp-2111/Desktop/final images/')

# if you want to show the images directly
# scan.show_images(images, names)

# if you want to save images
scan.save_images(images, names)

cv2.imshow('scan.jpg', scanned_image)
cv2.imshow('scanned binary.jpg', binary_output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# I suggest you use a different folder in a similar path from here
os.chdir('c:/Users/hp-2111/Desktop/final features/')

# changing brightness example
new1 = scan.change_brightness_contrast(scanned_image, 0, 1.25)
cv2.imwrite('contrast change.jpg', new1)
new1 = scan.change_brightness_contrast(scanned_image, 50, 1)
cv2.imwrite('brightness change.jpg', new1)

# changing HSV example
new1 = scan.change_HSV(scanned_image, h_offset=20, s_offset=0, v_offset=0)
cv2.imwrite('hsv change1.jpg', new1)
new1 = scan.change_HSV(scanned_image, h_offset=0, s_offset=0, v_offset=30)
cv2.imwrite('hsv change2.jpg', new1)

# resizing example
new1 = scan.change_size(scanned_image, 1)
cv2.imwrite('size change low.jpg', new1)
new1 = scan.change_size(scanned_image, 2)
cv2.imwrite('size change medium.jpg', new1)
new1 = scan.change_size(scanned_image, 3)
cv2.imwrite('size change optimal.jpg', new1)
cv2.imwrite('scanned image.jpg', scanned_image)
