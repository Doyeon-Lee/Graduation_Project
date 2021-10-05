import pickle
import datetime
import numpy as np
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

    with open("../model/first_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    X1 = np.concatenate((incli_arm_left, angle_arm_left), axis=1)
    X2 = np.concatenate((incli_leg_left, angle_leg_left), axis=1)
    X_left = np.vstack((X1, X2))
    ids_left = kmeans.predict(X_left)
    center = kmeans.cluster_centers_
    label_left = 1 if center[0][0] < center[1][0] else 0

    X1 = np.concatenate((incli_arm_right, angle_arm_right), axis=1)
    X2 = np.concatenate((incli_leg_right, angle_leg_right), axis=1)
    X_right = np.vstack((X1, X2))
    ids_right = kmeans.predict(X_right)
    center = kmeans.cluster_centers_
    label_right = 1 if center[0][0] < center[1][0] else 0

    violence_index = []
    violence_x = [[], []]
    violence_y = [[], []]
    non_violence_x = [[], []]
    non_violence_y = [[], []]
    # x와 y 나누기
    incli = incli_arm[0].copy()
    incli.extend(incli_leg[0])
    angle = angle_arm[0].copy()
    angle.extend(angle_leg[0])
    index, violence_x[0], violence_y[0], non_violence_x[0], non_violence_y[0] = \
        cluster_xy(label_left, ids_left, incli, angle, len(incli_arm[0]))
    violence_index.extend(index)

    incli = incli_arm[1].copy()
    incli.extend(incli_leg[1])
    angle = angle_arm[1].copy()
    angle.extend(angle_leg[1])
    index, violence_x[1], violence_y[1], non_violence_x[1], non_violence_y[1] = \
        cluster_xy(label_right, ids_right, incli, angle, len(incli_arm[1]))
    violence_index.extend(index)

    # plotting
    plt.subplot(1, 2, 1)
    plt.title("left")
    plt.xlabel('inclination variance')
    plt.ylabel('angle variance')
    plt.scatter(violence_x[0], violence_y[0], label='violence')
    plt.scatter(non_violence_x[0], non_violence_y[0], label='non-violence')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("right")
    plt.xlabel('inclination variance')
    plt.ylabel('angle variance')
    plt.scatter(violence_x[1], violence_y[1], label='violence')
    plt.scatter(non_violence_x[1], non_violence_y[1], label='non-violence')
    plt.legend()
    plt.show()

    # 폭력 의심 시간을 list로 뽑아서 저장
    time_list = []
    i = 0
    while i < len(violence_index):
        hour = violence_index[i] // 25 // 60 // 60
        minute = violence_index[i] // 25 // 60 % 60
        second = violence_index[i] // 25 % 60

        t = str(datetime.datetime.strptime(f'{hour}:{minute}:{second}', '%H:%M:%S').time())
        # 같은 의심 시간이 list에 있으면 list에 추가하지 않음
        if t not in time_list:
            time_list.append(t)
        # 연속적으로 폭력이 의심되면 가장 앞쪽 시간만 기록
        while i < (len(violence_index) - 1) and violence_index[i + 1] == violence_index[i] + 1:
            i += 1
        i += 1
    time_list.sort()
    return time_list
