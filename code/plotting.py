import math
import json
import numpy as np
import matplotlib.pyplot as plt

from rw_json import *
from child_distinguish import child_distinguish

accuracy_value = 0.3


# 세 점 사이의 각도 구하기
def angle_three_points(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = math.atan2(x1 - x2, y1 - y2)
    deg2 = math.atan2(x3 - x2, y3 - y2)

    deg1 = abs((deg1 * 180) / math.pi)  # 각도로 변환
    deg2 = abs((deg2 * 180) / math.pi)  # 각도로 변환
    res_deg = deg1 - deg2
    return res_deg if res_deg > 180 else res_deg-180


# 벡터로 변환
def make_vector(p1, p2):
    ax, ay = p1
    bx, by = p2
    return [bx - ax, by - ay]


# 직전/현재 기울기 벡터를 이용해 사이각 구하기
def get_incl_angle(v1, v2):
    innerAB = np.dot(v1, v2)
    AB = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(innerAB/AB)

    return angle / np.pi * 180


# 한 사람의 모든 관절의 (x, y, 정확도) 쌍을 리스트로 반환
def get_point_list(person):
    tmp = []
    for i in range(15):
        bt = person[BODY_POINT[i]]
        tmp.append((float(bt['x']), float(bt['y']), float(bt['accuracy'])))
    return np.asarray(tmp)


# specific(추적할 객체)과 person(관측된 관절/사람)의 좌표 거리 차이
def person_distance(specific_bt, person_bt):
    dist = 0
    num_cnt = 0
    for i in range(15):
        if specific_bt[i][2] < 0.5 or person_bt[i][2] < 0.5:
            continue

        dist += abs(specific_bt[i][0] - person_bt[i][0])
        dist += abs(specific_bt[i][1] - person_bt[i][1])
        num_cnt += 1

    if num_cnt is 0: return 0
    return dist / num_cnt


# 기울기 구하기
def get_incl(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    return abs(y1-y2) / abs(x1-x2)


# 특정 프레임수 단위로 끊어 그 값들을 평균을 낸다
def get_avg(val_list):
    num_avg = 10
    tmp = []
    s = len(val_list)
    iter = s // num_avg

    for i in range(iter):
        val = i*num_avg
        tmp.append(np.sum(val_list[val:val + num_avg]) / num_avg)

    # num_avg개씩 계산하고 남은 나머지 덩어리
    if s % num_avg > 0:
        tmp.extend(val_list[num_avg * iter:])

    return tmp


# 0인 값 ok
# 너무 큰 값이 나왔을때(정확도가 낮은 경우) 0.5 ok?
# 사람별 변화값 분리 필요

# skeleton json 에서 변화량 추출하기
def calc_variance():
    # 폭력 데이터
    for i in range(1, 116):
        variance_data = []
        SKELETON_FILENAME = V_SKELETON_FILEPATH + str(i) + ".json"
        PREPROCESSED_FILENAME = V_PREPROCESSED_FILEPATH + str(i) + ".json"

        # 하나의 관절 json 파일에 대하여
        try:
            with open(SKELETON_FILENAME) as f:
                json_data = json.load(f)
        except:
            continue

        num_people = len(json_data[0]['person'])

        # 0번째 프레임에 있는 사람 수만큼 반복한다
        for person_id in range(num_people):
            variance_data.append(make_preprocessed_json(SKELETON_FILENAME, person_id))

        with open(PREPROCESSED_FILENAME, 'w', encoding="utf-8") as make_file:
            json.dump(variance_data, make_file, ensure_ascii=False, indent="\t")

    # 비폭력 데이터 cam1(다른 파일의 경우 global_data 파일의 경로를 수정해주자!)
    for i in range(1, 61):
        variance_data = []
        SKELETON_FILENAME = NONV_SKELETON_FILEPATH + str(i) + ".json"
        PREPROCESSED_FILENAME = NONV_PREPROCESSED_FILEPATH + str(i) + ".json"

        # 하나의 관절 json 파일에 대하여
        with open(SKELETON_FILENAME) as f:
            json_data = json.load(f)
            num_people = len(json_data[0]['person'])

            # 0번째 프레임에 있는 사람 수만큼 반복한다
            for person_id in range(num_people):
                variance_data.append(make_preprocessed_json(SKELETON_FILENAME, person_id))

            with open(PREPROCESSED_FILENAME, 'w', encoding="utf-8") as make_file:
                json.dump(variance_data, make_file, ensure_ascii=False, indent="\t")


def make_preprocessed_json(json_filename, person_id):
    angle_list = []
    incl_list = []

    result = {'person_id': person_id}
    variance_list = []

    # 관심 있는 관절의 기울기와 각도를 추출
    for point_number in range(4):
        # tmp_angle, tmp_incl = get_variance(json_filename, person_id, point_number)
        # angle_list.extend(tmp_angle)
        # incl_list.extend(tmp_incl)
        angle_list, incl_list = get_variance(json_filename, person_id, point_number)

        # 팔다리 하나씩의 변화량
        variance = {
            "angle_variance": angle_list,
            "incl_variance": incl_list
        }
        point = {SPECIFIC_JOINT[point_number]: variance}
        variance_list.append(point) # 4가지 모두 저장

    result["variance"] = variance_list
    return result


# person id는 plotting 결과를 얻고자 함수를 반복하기 위해 임의로 넣은 것
def get_variance(json_filename, point_number):
    global accuracy_value
    with open(json_filename) as f:
        json_data = json.load(f)
        json_len = len(json_data)
        angle_list = []
        incl_list = []

        prev_frame_num = json_data[0]["frame_id"] # 마지막에 관절이 인식된 프레임 번호(3개 점중 하나라도 0이거나 신뢰도가 낮으면)
        pre_list = [0.0, [0, 0]] # 해당 관절에 대한 직전 각도와 (수학적)벡터값 저장
        adult_id = child_distinguish(0, "", json_data)
        specific = json_data[0]['person'][adult_id]['keypoint']  # 우리가 원하는 특정한 객체 specific!
        specific_bt = get_point_list(specific)

        # 모든 프레임마다 반복하며
        for i in range(0, json_len):
            # print(f"\nframe = {i}")
            obj = json_data[i]['person']
            obj_len = len(obj)

            dist_list = []
            # 모든 사람을 탐색해서 우리가 찾고자 하는 객체가 맞는지 검사
            for j in range(obj_len):
                person = obj[j]['keypoint']
                # 관절값 리스트로 저장
                person_bt = get_point_list(person)
                dist_list.append(person_distance(specific_bt, person_bt))

            # 관절들의 거리의 합의 평균이 가장 적은게 같은 대상이라 판단
            # 한 명만 인식됐을 경우, 그 객체가 맞는지 검사
            dist_list = np.array(dist_list)
            specific_id = dist_list.argmin()

            # 머리와 두 어깨의 인식률이 모두 안좋을 땐 continue
            if specific["Head"]['accuracy'] < 0.3 and specific["RShoulder"]['accuracy'] < 0.3 and \
                    specific["LShoulder"]['accuracy'] < 0.3:
                continue

            head = specific["Head"]
            if specific["RShoulder"]['accuracy'] < accuracy_value:
                shoulder = specific["LShoulder"]
            elif specific["LShoulder"]['accuracy'] < accuracy_value:
                shoulder = specific["RShoulder"]
            else:
                # 어깨의 가운데 점을 사용
                shoulder = {'x': (specific["RShoulder"]['x'] + specific["LShoulder"]['x']) / 2, \
                            'y': (specific["RShoulder"]['y'] + specific["LShoulder"]['y']) / 2}
            head_len = ((head['x'] - shoulder['x']) ** 2 + (head['y'] - shoulder['y']) ** 2) ** 0.5

            w, h = get_frame_size()
            skipped_frame_num = json_data[i]["frame_id"] - prev_frame_num   # 현재 프레임과의 차이

            max_range = ((head_len**2 * 56) / (165 * h)) * skipped_frame_num
            if dist_list[specific_id] < max_range:
                specific = obj[specific_id]['keypoint']
                specific_bt = get_point_list(specific)
            else:
                continue

            # 추적하고자 하는 객체의 관절 변화를 계산
            p1, p2, p3 = SPECIFIC_POINT[point_number]

            # 추적하는 객체의 관절값 가져오기
            p1 = specific[p1]; p2 = specific[p2]; p3 = specific[p3]
            acc1 = float(p1['accuracy']); acc2 = float(p2['accuracy']); acc3 = float(p3['accuracy'])
            p1 = (p1['x'], p1['y']); p2 = (p2['x'], p2['y']); p3 = (p3['x'], p3['y'])

            # 0이 있으면 해당 frame 건너 뛴다
            if acc1 < 0.3 or acc2 < 0.3 or acc3 < 0.3:
                continue

            # 세 점 사이의 각도를 구하여 직전 각도의 차이를 기록한다
            cur_angle = angle_three_points(p1, p2, p3)
            # 두 점을 잇는 벡터와 직전 벡터와의 각도를 구한 후, 기록한다
            cur_vec = make_vector(p1, p3)

            if pre_list != [0.0, [0, 0]]:
                if skipped_frame_num > 0:
                    # 맨 앞과 맨 뒤를 더해줌
                    skipped_frame_num += 1
                    # 넘어간 프레임을 채워줌
                    missing_angle_list = np.linspace(pre_list[0], cur_angle, skipped_frame_num)
                    missing_incl_list = np.linspace(pre_list[1], cur_vec, skipped_frame_num)
                    j = 1
                    while j < skipped_frame_num:
                        sub_angle = abs(missing_angle_list[j] - missing_angle_list[j - 1])
                        angle_list.append(sub_angle)

                        sub_incl_angle = get_incl_angle(missing_incl_list[j], missing_incl_list[j - 1])
                        incl_list.append(sub_incl_angle)
                        j += 1
                else:
                    sub_angle = abs(pre_list[0] - cur_angle)
                    angle_list.append(sub_angle)

                    sub_incl_angle = get_incl_angle(cur_vec, pre_list[1])
                    incl_list.append(sub_incl_angle)
            pre_list[0] = cur_angle
            pre_list[1] = cur_vec

            # 여기까지 왔으면 frame 안넘어갔겠다
            prev_frame_num = json_data[i]["frame_id"]

    # n개의 frame을 단위로 그 값을 평균을 낸다 => 1초가 몇 frame인지 고려하면 좋을 듯
    # avg_angle = get_avg(angle_list)
    # avg_incl = get_avg(incl_list)
    # return avg_angle, avg_incl
    return angle_list, incl_list


def plotting_points():
    # 값 변화 plotting 하기
    non_violence_json_file = [str(i) for i in range(1, 61)]
    violence_json_file = [str(i) for i in range(1, 116)]

    # plotting에 쓸 리스트
    angle_list = []
    incl_list = []
    # 폭력 데이터셋
    for name in violence_json_file:
        try:
            with open(V_PREPROCESSED_FILEPATH + name + '.json') as f:
                json_data = json.load(f)
                element = json_data[0]['variance'][0]  # LArm의 변화값
                angle_list.extend(element['LArm']['angle_variance'])
                incl_list.extend(element['LArm']['incl_variance'])
        except:
            continue

    plt.scatter(incl_list, angle_list, label="violence")

    # 일정 이상의 수치가 나오는 놈들 출력
    l = len(angle_list)
    for i in range(l):
        if angle_list[i] > 50 or incl_list[i] > 100:
            print(angle_list[i], incl_list[i])

    # 다른 색으로 적용하기 위해 리스트 다시 초기화
    angle_list = []
    incl_list = []
    # 비폭력 데이터셋
    for name in non_violence_json_file:
        with open(NONV_PREPROCESSED_FILEPATH + name + '.json') as f:
            json_data = json.load(f)
            element = json_data[0]['variance'][0] #LArm의 변화값
            angle_list.extend(element['LArm']['angle_variance'])
            incl_list.extend(element['LArm']['incl_variance'])

    plt.scatter(incl_list, angle_list, c='red', label="non-violence")

    # 일정 이상의 수치가 나오는 놈들 출력
    l = len(angle_list)
    for i in range(l):
        if angle_list[i] > 50 or incl_list[i] > 100:
            print(angle_list[i], incl_list[i])

    plt.xlabel('inclination variance')
    plt.ylabel('angle variance')
    plt.legend()
    plt.show()


def get_distance(json_obj, skeleton_id):
    distance = 0
    key_count = 0
    adult_obj = get_prev_adult_point()
    for key in BODY_POINT[:-1]:
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

