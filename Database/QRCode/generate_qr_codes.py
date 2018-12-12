import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import numpy as np

import qrcode
img = qrcode.make('Some data here')

print('img = {}'.format(img))

img.show()



# image = np.asarray(img)
#
# print('image = {}'.format(image))

# cv2.imshow('qrcode', image)
# cv2.waitKey(-1)