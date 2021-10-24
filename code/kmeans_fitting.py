import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from plotting import get_variance


def clustering(kmeans, skeleton_json_file):
    # 관절들의 변화량을 list로 저장
    # 0은 left, 1은 right
    angle_arm = [[], []]
    incli_arm = [[], []]
    angle_leg = [[], []]
    incli_leg = [[], []]
    for i in range(4):
        angle, incli = get_variance(skeleton_json_file, i)
        if i is 0:
            angle_arm[0].extend(angle)
            incli_arm[0].extend(incli)
        elif i is 1:
            angle_arm[1].extend(angle)
            incli_arm[1].extend(incli)
        elif i is 2:
            angle_leg[0].extend(angle)
            incli_leg[0].extend(incli)
        elif i is 3:
            angle_leg[1].extend(angle)
            incli_leg[1].extend(incli)

    incli_arm_left = [[]]; angle_arm_left = [[]]
    incli_leg_left = [[]]; angle_leg_left = [[]]
    incli_arm_right = [[]]; angle_arm_right = [[]]
    incli_leg_right = [[]]; angle_leg_right = [[]]
    # clustering
    if len(incli_arm[0]) > 0:
        incli_arm_left = np.array(incli_arm[0]).reshape(len(incli_arm[0]), -1)
    if len(angle_arm[0]) > 0:
        angle_arm_left = np.array(angle_arm[0]).reshape(len(angle_arm[0]), -1)
    if len(incli_leg[0]) > 0:
        incli_leg_left = np.array(incli_leg[0]).reshape(len(incli_leg[0]), -1)
    if len(angle_leg[0]) > 0:
        angle_leg_left = np.array(angle_leg[0]).reshape(len(angle_leg[0]), -1)

    if len(incli_arm[1]) > 0:
        incli_arm_right = np.array(incli_arm[1]).reshape(len(incli_arm[1]), -1)
    if len(angle_arm[1]) > 0:
        angle_arm_right = np.array(angle_arm[1]).reshape(len(angle_arm[1]), -1)
    if len(incli_leg[1]) > 0:
        incli_leg_right = np.array(incli_leg[1]).reshape(len(incli_leg[1]), -1)
    if len(angle_leg[1]) > 0:
        angle_leg_right = np.array(angle_leg[1]).reshape(len(angle_leg[1]), -1)

    X1 = np.concatenate((incli_arm_left, angle_arm_left), axis=1)
    X2 = np.concatenate((incli_leg_left, angle_leg_left), axis=1)

    # fitting
    X_left = [[]]
    if X2.shape[1] > 0:
        if X1.shape[1] > 0:
            X_left = np.vstack((X1, X2))
        else:
            X_left = X2
    else:
        if X1.shape[1] > 0:
            X_left = X1

    # nan 제거
    X_left = pd.DataFrame(X_left)
    X_left = X_left.dropna(how="any")
    X_left = X_left.to_numpy()

    if X_left.shape[1] > 0:
        kmeans.fit(X_left)

    X1 = np.concatenate((incli_arm_right, angle_arm_right), axis=1)
    X2 = np.concatenate((incli_leg_right, angle_leg_right), axis=1)

    # fitting
    X_right = [[]]
    if X2.shape[1] > 0:
        if X1.shape[1] > 0:
            X_right = np.vstack((X1, X2))
        else:
            X_right = X2
    else:
        if X1.shape[1] > 0:
            X_right = X1

    # nan 제거
    X_right = pd.DataFrame(X_right)
    X_right = X_right.dropna(how="any")
    X_right = X_right.to_numpy()

    if X_right.shape[1] > 0:
        kmeans.fit(X_right)

    return kmeans


# violence_list = os.listdir('../media/violence')
# violence_list = [re.sub('.mp4', '', i) for i in violence_list]

# non_violence_list = os.listdir('../media/non-violence')
# non_violence_list = [re.sub('.mp4', '', i) for i in non_violence_list]

non_violence_list = ["f2", "601", "904", "1001", "119", "120", "220", "306", "305", "218", "114"]

with open("../model/sv_model_violence.pkl", "rb") as f:
    kmeans = pickle.load(f)

# kmeans = KMeans(n_clusters=2)

# for i in violence_list:
#     skeleton_json_file = f'../output/video/{i}/results{i}.json'
#     kmeans = clustering(kmeans, skeleton_json_file)

for i in non_violence_list:
    skeleton_json_file = f'../output/video/{i}/results{i}.json'
    kmeans = clustering(kmeans, skeleton_json_file)

with open("../model/sv_model18.pkl", "wb") as f:
    pickle.dump(kmeans, f)
