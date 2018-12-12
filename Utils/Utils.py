import numpy as np # linear algebra

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['figure.figsize'] = (12, 12)

# def plot_samples(face_images, facial_keypoints):
#
#     # show a random subset of images from the dataset
#
#     num_fig_rows = 4
#     num_fig_cols = 4
#
#     num_plots = num_fig_rows * num_fig_cols\
#
#     print('face_images.shape = {}'.format(face_images.shape))
#
#     rand_inds_vec = np.random.choice(face_images.shape[1], num_plots, replace=False)
#     rand_inds_mat = rand_inds_vec.reshape((num_fig_rows, num_fig_cols))
#
#     plt.close('all')
#     fig, ax = plt.subplots(nrows=num_fig_rows, ncols=num_fig_cols, figsize=(14, 18))
#
#     for i in range(num_fig_rows):
#         for j in range(num_fig_cols):
#             curr_ind = rand_inds_mat[i][j]
#             curr_image = face_images[curr_ind, :, :]*255.0
#
#             x_feature_coords = np.array(facial_keypoints.iloc[curr_ind, 0::2].tolist())
#             y_feature_coords = np.array(facial_keypoints.iloc[curr_ind, 1::2].tolist())
#
#             ax[i][j].imshow(curr_image, cmap='gray')
#             ax[i][j].scatter(x_feature_coords, y_feature_coords, c='r', s=15)
#             ax[i][j].set_axis_off()
#             ax[i][j].set_title('image index = %d' % (curr_ind), fontsize=10)
#
#     plt.show()


def plot_samples(X, Y):
    print('X.shape = {}'.format(X.shape))

    Spic = X.shape[1]
    n = 0
    nrows = 4
    ncols = 4
    irand=np.random.choice(Y.shape[0],nrows*ncols)
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=[ncols*2,nrows*2])
    for row in range(nrows):
        for col in range(ncols):
            ax[row,col].imshow(X[irand[n],:,:,0], cmap='gray')
            ax[row,col].scatter(Y[irand[n],0::2]*Spic,Y[irand[n],1::2]*Spic,marker='X',c='r',s=100)
            ax[row,col].set_xticks(())
            ax[row,col].set_yticks(())
            ax[row,col].set_title('image index = %d' %(irand[n]),fontsize=10)
            n += 1

    plt.show()

import random

def split_set(X, y, valid=0.2):
    # Xt, yt, Xv, yv = [], [], [], []
    #
    len = X.shape[0]
    train_len = int(len * (1-valid))
    #
    # for idx in range(len):
    #     # user_id = filename[-21:-13]
    #     # random.seed(user_id)
    #     if idx > train_len:
    #         Xv.append(X[idx])
    #         yv.append(y[idx])
    #     else:
    #         Xt.append(X[idx])
    #         yt.append(y[idx])

    # assert len(X) == len(Xt) + len(Xv)
    Xt, Xv = np.split(X, [train_len])
    yt, yv = np.split(y, [train_len])

    return Xt, yt, Xv, yv