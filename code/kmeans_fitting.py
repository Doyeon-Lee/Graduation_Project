import os
import re
import cv2
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from plotting import get_variance
from global_data import set_frame_size


def clustering(kmeans, skeleton_json_file):
    # 관절들의 변화량을 list로 저장
    angle_arm = []
    incli_arm = []
    angle_leg = []
    incli_leg = []
    for i in range(4):
        angle, incli = get_variance(skeleton_json_file, i)
        if i < 2:
            angle_arm.extend(angle)
            incli_arm.extend(incli)
        else:
            angle_leg.extend(angle)
            incli_leg.extend(incli)

    incli_arm_ = [[]]
    incli_leg_ = [[]]
    angle_arm_ = [[]]
    angle_leg_ = [[]]
    # clustering
    if len(incli_arm) > 0:
        incli_arm_ = np.array(incli_arm).reshape(len(incli_arm), -1)
    if len(angle_arm) > 0:
        angle_arm_ = np.array(angle_arm).reshape(len(angle_arm), -1)
    if len(incli_leg) > 0:
        incli_leg_ = np.array(incli_leg).reshape(len(incli_leg), -1)
    if len(angle_leg) > 0:
        angle_leg_ = np.array(angle_leg).reshape(len(angle_leg), -1)

    X1 = np.concatenate((incli_arm_, angle_arm_), axis=1)
    X2 = np.concatenate((incli_leg_, angle_leg_), axis=1)

    # fitting
    X = [[]]
    if X2.shape[1] > 0:
        if X1.shape[1] > 0:
            X = np.vstack((X1, X2))
        else:
            X = X2
    else:
        if X1.shape[1] > 0:
            X = X1

    # nan 제거
    X = pd.DataFrame(X)
    X = X.dropna(how="any")
    X = X.to_numpy()

    if X.shape[1] > 0:
        kmeans.fit(X)

    return kmeans


violence_list = os.listdir('../media/violence')
violence_list = [re.sub('.mp4', '', i) for i in violence_list]

# non_violence_list = os.listdir('../media/non-violence')
# non_violence_list = [re.sub('.mp4', '', i) for i in non_violence_list]

# violence_list = []
# for i in range(1, 48):
#     violence_list.append('m'+str(i))

# with open("../model/sv_model_non_violence.pkl", "rb") as f:
#     kmeans = pickle.load(f)

kmeans = KMeans(n_clusters=2)

for i in violence_list:
    skeleton_json_file = f'../output/video/{i}/results{i}.json'
    path = f'../media/violence/{i}.mp4'

    # frame_size 정해줌
    cap = cv2.VideoCapture(path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    set_frame_size(int(width), int(height))
    cap.release()

    kmeans = clustering(kmeans, skeleton_json_file)

# for i in non_violence_list:
#     skeleton_json_file = f'../output/video/{i}/results{i}.json'
#     path = f'../media/non-violence/{i}.mp4'
#
#     # frame_size 정해줌
#     cap = cv2.VideoCapture(path)
#     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     set_frame_size(int(width), int(height))
#     cap.release()
#
#     kmeans = clustering(kmeans, skeleton_json_file)

with open("../model/sv_model_v.pkl", "wb") as f:
    pickle.dump(kmeans, f)
