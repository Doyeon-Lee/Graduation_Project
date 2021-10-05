import pickle
import numpy as np
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

    # clustering
    incli_arm_left = np.array(incli_arm[0]).reshape(len(incli_arm[0]), -1)
    angle_arm_left = np.array(angle_arm[0]).reshape(len(angle_arm[0]), -1)
    incli_leg_left = np.array(incli_leg[0]).reshape(len(incli_leg[0]), -1)
    angle_leg_left = np.array(angle_leg[0]).reshape(len(angle_leg[0]), -1)

    incli_arm_right = np.array(incli_arm[1]).reshape(len(incli_arm[1]), -1)
    angle_arm_right = np.array(angle_arm[1]).reshape(len(angle_arm[1]), -1)
    incli_leg_right = np.array(incli_leg[1]).reshape(len(incli_leg[1]), -1)
    angle_leg_right = np.array(angle_leg[1]).reshape(len(angle_leg[1]), -1)

    X1 = np.concatenate((incli_arm_left, angle_arm_left), axis=1)
    X2 = np.concatenate((incli_leg_left, angle_leg_left), axis=1)
    X_left = np.vstack((X1, X2))
    kmeans.fit(X_left)

    X1 = np.concatenate((incli_arm_right, angle_arm_right), axis=1)
    X2 = np.concatenate((incli_leg_right, angle_leg_right), axis=1)
    X_right = np.vstack((X1, X2))
    kmeans.fit(X_right)


file_list = ["218", "p1"]
kmeans = KMeans(n_clusters=2)

for i in file_list:
    skeleton_json_file = f'../output/video/{i}/results{i}.json'
    clustering(kmeans, skeleton_json_file)

with open("../model/sv_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)
