# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# import Utils as utils

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "./"]).decode("utf8"))

def get_data(image_path, landmark_path):
    # load the dataset
    face_images_db = np.load(image_path)['face_images']
    facial_keypoints_df = pd.read_csv(landmark_path)

    (im_height, im_width, num_images) = face_images_db.shape
    num_keypoints = facial_keypoints_df.shape[1] / 2

    print('number of images = %d' % (num_images))
    print('image dimentions = (%d,%d)' % (im_height, im_width))
    print('number of facial keypoints = %d' % (num_keypoints))

    # Filter database

    # 原始数据图像是 [cols, rows, idx] 现在将最后一维挪到位置0，变成 [idx, cols, rows]
    face_images = np.moveaxis(face_images_db, -1, 0)

    # 只筛选所需要的几个数据不为 0 的部分
    iselect = \
    np.nonzero(facial_keypoints_df.left_eye_center_x.notna() & facial_keypoints_df.right_eye_center_x.notna() &
               facial_keypoints_df.nose_tip_x.notna() & facial_keypoints_df.mouth_center_bottom_lip_x.notna())[0]

    Spic = face_images.shape[1]
    m = iselect.shape[0]
    X = np.zeros((m, Spic, Spic, 1))
    Y = np.zeros((m, 8))

    # X 表示归一化图像，扩展表示batchsize = 1
    X[:, :, :, 0] = face_images[iselect, :, :] / 255.0

    # Y 表示 Label，只做左右眼和鼻子\嘴,总共5个关键点，真值变成比例
    Y[:, 0] = facial_keypoints_df.left_eye_center_x[iselect] / Spic
    Y[:, 1] = facial_keypoints_df.left_eye_center_y[iselect] / Spic
    Y[:, 2] = facial_keypoints_df.right_eye_center_x[iselect] / Spic
    Y[:, 3] = facial_keypoints_df.right_eye_center_y[iselect] / Spic
    Y[:, 4] = facial_keypoints_df.nose_tip_x[iselect] / Spic
    Y[:, 5] = facial_keypoints_df.nose_tip_y[iselect] / Spic
    Y[:, 6] = facial_keypoints_df.mouth_center_bottom_lip_x[iselect] / Spic
    Y[:, 7] = facial_keypoints_df.mouth_center_bottom_lip_y[iselect] / Spic

    print('# selected images = %d' % (m))

    # 显示并绘制数据
    # utils.plot_samples(X, Y)

    return X, Y