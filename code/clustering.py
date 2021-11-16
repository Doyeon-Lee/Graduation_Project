import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotting import get_variance
from global_data import set_frame_num, get_frame_num


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


def make_pair(start_frame, end_frame):
    start_video_frame = max(start_frame - 30, 0)
    end_video_frame = min(get_frame_num(), start_frame + 30 if end_frame == -1 else end_frame + 30)
    return [start_video_frame, end_video_frame]


def make_time_pair(violence_index):
    # violence_index가 비어있으면 바로 return
    if len(violence_index) == 0:
        return []

    # 앞 뒤로 2초내에 벌어진 상황이면 같은 영상으로 편집(30fps라고 가정)
    one_video_frame = 60
    time_pairs = []

    start_frame = violence_index[0]
    end_frame = -1

    for cand_idx in violence_index:
        if end_frame == -1:
            end_frame = start_frame

        if end_frame + one_video_frame < cand_idx:
            time_pair = make_pair(start_frame, end_frame)
            time_pairs.append(time_pair)
            start_frame = cand_idx
            end_frame = -1
        else:
            end_frame = cand_idx

    # start만 있고 end는 없을 경우
    if start_frame != end_frame:
        time_pair = make_pair(start_frame, end_frame)
        time_pairs.append(time_pair)

    return time_pairs


def clustering(skeleton_json_file):
    # 영상의 frame 수를 저장
    with open(skeleton_json_file) as f:
        json_data = json.load(f)
    set_frame_num(int(json_data[-1]['frame_id']))

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
        # plt.show()

    # 폭력 의심 시간을 list로 뽑아서 저장
    violence_index.sort()
    time_list = make_time_pair(violence_index)
    return time_list
