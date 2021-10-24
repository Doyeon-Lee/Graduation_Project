import json
import numpy as np
from global_data import *


# 기울기 구하기
def get_incl(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    return abs(y1-y2) / abs(x1-x2)


def get_distance(json_obj, skeleton_id):
    distance = 0
    key_count = 0
    adult_obj = get_prev_adult_point()
    for key in body_point[:-1]:
        if json_obj[0]['person'][skeleton_id]['keypoint'][key]['accuracy'] >= 0.7:
            x = json_obj[0]['person'][skeleton_id]['keypoint'][key]['x']
            y = json_obj[0]['person'][skeleton_id]['keypoint'][key]['y']

            x2 = adult_obj[key]['x']
            y2 = adult_obj[key]['y']
            accuracy = adult_obj[key]['accuracy']

            # 정확도가 높으면 움직인 거리 계산
            if accuracy >= 0.7:
                key_count += 1
                distance += ((x - x2) ** 2 + (y - y2) ** 2) ** 0.5
    return distance, key_count


def child_distinguish(frame_num, file_name="", data=None):
    if data is None:
        data = [{}]
    if file_name != "":
        # json 파일 열기
        with open(file_name, 'r') as f:
            json_data = json.load(f)
    else:
        json_data = data

    body_ratio_list = np.array([])
    for person_num in range(0, len(json_data[0]['person'])):
        Head = json_data[0]['person'][person_num]['keypoint']['Head']
        Chest = json_data[0]['person'][person_num]['keypoint']['Chest']

        RShoulder = json_data[0]['person'][person_num]['keypoint']['RShoulder']
        LShoulder = json_data[0]['person'][person_num]['keypoint']['LShoulder']

        RHip = json_data[0]['person'][person_num]['keypoint']['RHip']
        LHip = json_data[0]['person'][person_num]['keypoint']['LHip']

        # 머리/어깨/가슴/엉덩이가 인식이 안되면 쓰레기값을 넣어주고 넘어감
        if Head['accuracy'] < 0.7 or (RShoulder['accuracy'] < 0.7 and LShoulder['accuracy'] < 0.7) or \
                Chest['accuracy'] < 0.7 or (RHip['accuracy'] < 0.7 and LHip['accuracy'] < 0.7):
            body_ratio_list = np.append(body_ratio_list, 1)
            continue

        if RShoulder['accuracy'] < 0.7:
            Shoulder = LShoulder
        elif LShoulder['accuracy'] < 0.7:
            Shoulder = RShoulder
        else:
            # 어깨의 가운데 점을 사용
            Shoulder = {'x': (RShoulder['x'] + LShoulder['x']) / 2, 'y': (RShoulder['y'] + LShoulder['y']) / 2}

        len_head = ((Head['x'] - Shoulder['x']) ** 2 + (Head['y'] - Shoulder['y']) ** 2) ** 0.5

        if RHip['accuracy'] < 0.7:
            Hip = LHip
        elif LHip['accuracy'] < 0.7:
            Hip = RHip
        else:
            # 엉덩이의 가운데 점을 사용
            Hip = {'x': (RHip['x'] + LHip['x']) / 2, 'y': (RHip['y'] + LHip['y']) / 2}

        len_body = ((Shoulder['x'] - Chest['x']) ** 2 + (Shoulder['y'] - Chest['y']) ** 2) ** 0.5 + \
                   ((Chest['x'] - Hip['x']) ** 2 + (Chest['y'] - Hip['y']) ** 2) ** 0.5

        if len_head == 0 or len_head + len_body == 0:
            body_ratio_list = np.append(body_ratio_list, 1)
            continue
        else:
            body_ratio = len_head / (len_head + len_body)

            # 허리를 굽혀 비율이 잘못 나왔다고 판단하여 머리 비율을 조정
            if get_incl((Head['x'], Head['y']), (Hip['x'], Hip['y'])) < 1:
                if body_ratio >= 0.44:
                    body_ratio *= 0.96
                elif body_ratio < 0.4:
                    body_ratio *= 1.1

            # 비율이 0.41보다 작으면 제대로 추출되지 않았다고 간주
            if body_ratio >= 0.41:
                body_ratio_list = np.append(body_ratio_list, body_ratio)
                if data == [{}]:
                    set_rate(body_ratio)
            else:
                body_ratio_list = np.append(body_ratio_list, 1)

    if len(body_ratio_list) == 0:
        return -1

    # 후보의 비율과 키값
    candidate_ratio = min(body_ratio_list)
    candidate_key = np.where(body_ratio_list == candidate_ratio)[0][0]

    # 쓰레기값(1) 제외
    while 1 in body_ratio_list:
        body_ratio_list = np.delete(body_ratio_list, np.where(body_ratio_list == 1))

    if len(body_ratio_list) == 0:
        return -1

    # 인원이 2명보다 많을 때, 성인으로 추정되는 사람들의 비율값들을 모은 리스트의 표준편차값과
    # 전체 인원의 표준편차값 중 더 작은 값을 표준편차값으로 사용
    if len(body_ratio_list) > 1:
        body_ratio_list.sort()
        list_for_slice = np.where(body_ratio_list >= 0.44)[0]
        if len(list_for_slice) > 0:
            num_for_slice = int(list_for_slice[0])
            if num_for_slice > 0:
                standard = min(np.std(body_ratio_list[:num_for_slice]), np.std(body_ratio_list))
            else:
                standard = np.std(body_ratio_list)
        else:
            standard = np.std(body_ratio_list)
    else:
        standard = 0

    ratio_sum, people = get_rate()
    if people == 0:
        average = 0
    else:
        average = ratio_sum / people

    # 인식되는 사람이 한 명일 때 prev_adult_point와 비교하여 거리값이 비슷하면 key값 return
    skeleton_list = get_skeleton_list()
    if standard == 0:
        if len(skeleton_list) > 0:
            distance, key_count = get_distance(json_data, candidate_key)
            w, h = get_frame_size()
            skipped_frame_num = frame_num - skeleton_list[-1]['frame_id']  # 현재 프레임과의 차이
            max_range = ((get_prev_adult_head_len()**2 * 56) / (165 * h)) * skipped_frame_num
            if key_count > 0 and distance / key_count < max_range:
                set_current_adult_point(json_data[0]['person'][candidate_key]['keypoint'])
                set_prev_adult_point(get_current_adult_point())
                return candidate_key
            else:
                return -1
        # 인식되는 사람이 한 명일 때 특정 범위 내의 비율을 가진 사람을 성인이라고 생각
        else:
            if 0.41 <= candidate_ratio < 0.44:
                set_current_adult_point(json_data[0]['person'][candidate_key]['keypoint'])
                return candidate_key
            else:
                return -1
    # 인식되는 사람이 두 명 이상일 때
    else:
        # 표준편차값이 0.0175보다 작으면 성인이 감지되지 않았다고 생각
        # 성인의 자세가 흐트러졌을 경우 관절을 잘못 인식할 수 있기 때문에 제외
        if standard < 0.0175:
            return -1
        else:
            # 성인이 많이 인식되어 평균값이 성인에 가까울 때
            if average < 0.45:
                if abs(average - candidate_ratio) <= 0.024:
                    set_current_adult_point(json_data[0]['person'][candidate_key]['keypoint'])
                    return candidate_key
            # 아이가 많이 인식되어 평균값이 아이에 가까울 때
            else:
                if average - candidate_ratio >= 0.03:
                    set_current_adult_point(json_data[0]['person'][candidate_key]['keypoint'])
                    return candidate_key

    # 성인이 없다면 -1을 return
    return -1
