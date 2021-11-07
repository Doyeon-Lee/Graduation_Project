import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from plotting import get_variance


# x와 y 나누기
def cluster_xy(label, ids, incli_list, angle_list, len_arm):
    # 팔과 다리의 시작 frame 번호를 index 변수로 둠
    index = 0
    violence_index = []
    violence_x = []; violence_y = []
    non_violence_x = []; non_violence_y = []
    for i in range(len(ids)):
        # 다리 시작
        if i == len_arm:
            index = 0
        if ids[i] == label:
            violence_index.append(index)
            violence_x.append(incli_list[i])
            violence_y.append(angle_list[i])
        else:
            non_violence_x.append(incli_list[i])
            non_violence_y.append(angle_list[i])
        index += 1
    return violence_index, violence_x, violence_y, non_violence_x, non_violence_y


def clustering(skeleton_json_file):
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

    with open("../model/sv_model_v.pkl", "rb") as f:
        kmeans = pickle.load(f)
    X1 = np.concatenate((incli_arm_, angle_arm_), axis=1)
    X2 = np.concatenate((incli_leg_, angle_leg_), axis=1)

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

    is_available = False
    if X.shape[1] > 0:
        ids = kmeans.predict(X)
        center = kmeans.cluster_centers_
        label = 1 if center[0][0] < center[1][0] else 0
        is_available = True

    violence_index = []
    violence_x = []
    violence_y = []
    non_violence_x = []
    non_violence_y = []
    # x와 y 나누기
    if is_available:
        incli = incli_arm.copy()
        incli.extend(incli_leg)
        angle = angle_arm.copy()
        angle.extend(angle_leg)
        index, violence_x, violence_y, non_violence_x, non_violence_y = \
            cluster_xy(label, ids, incli, angle, len(incli_arm))
        violence_index.extend(index)

    # plotting
    if is_available:
        plt.xlabel('inclination variance')
        plt.ylabel('angle variance')
        plt.scatter(violence_x, violence_y, label='violence')
        plt.scatter(non_violence_x, non_violence_y, label='non-violence')
        plt.legend()
        plt.show()

    # 폭력 의심 시간을 list로 뽑아서 저장
    # frame을 초로 변환 후 중복 제거
    violence_index = [i // 25 for i in violence_index]
    violence_index = list(set(violence_index))
    violence_index.sort()
    time_list = []
    recorded_sec = -2
    i = 0
    while i < len(violence_index):
        # 연속적으로 폭력이 의심되면 가장 앞쪽 시간만 기록
        sec = violence_index[i]
        if recorded_sec + 1 == sec:
            recorded_sec = sec
            i += 1
            continue

        hour = violence_index[i] // 60 // 60
        minute = violence_index[i] // 60 % 60
        second = violence_index[i] % 60

        t = str(datetime.datetime.strptime(f'{hour}:{minute}:{second}', '%H:%M:%S').time())
        # 같은 의심 시간이 list에 있으면 list에 추가하지 않음
        if t not in time_list:
            time_list.append(t)
            recorded_sec = sec
        i += 1
    return time_list
