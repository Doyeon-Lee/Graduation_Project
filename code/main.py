# from skeleton import *  # import하면 wrapper 가 실행된다
from video_tracking import *
from rw_json import *
import math


# def check_boundary(person_bodypoint, frame_w, frame_h):
#     n = body_dict['Neck']
#     ls = body_dict['LShoulder']
#     rs = body_dict['RShoulder']
#     mh = body_dict['MidHip']
#     lh = body_dict['LHip']
#     rh = body_dict['RHip']
#
#     shoulder_len = max(person_bodypoint[n][0] - person_bodypoint[rs][0],
#                        person_bodypoint[n][0] - person_bodypoint[ls][0])
#
#     _x = max(0, int(person_bodypoint[n][0] - shoulder_len))
#     _y = int(person_bodypoint[n][1])
#     _w = 0
#     _h = 0
#
#     # MidHip이 안보일 경우
#     if not person_bodypoint[mh][1]:
#         _h = int(frame_h - person_bodypoint[n][1])
#     else:
#         _h = int(person_bodypoint[mh][1] - person_bodypoint[n][1])
#
#     # Neck == 0
#     if not person_bodypoint[n][0]:
#         if not person_bodypoint[lh][0]:  # LHip == 0
#             _w = min(person_bodypoint[rh][0], FRAME_W - person_bodypoint[rh][0])
#         elif not person_bodypoint[rh][0]:
#             _w = min(person_bodypoint[lh][0], FRAME_W - person_bodypoint[lh][0])
#         else:
#             _w = abs(person_bodypoint[rh][0] - person_bodypoint[lh][0])
#
#     else:  # Neck != 0
#         if not person_bodypoint[mh][0]:  # MidHip == 0
#             if not person_bodypoint[ls][0]:  # LShoulder == 0
#                 _w = min(person_bodypoint[rs][0], FRAME_W - person_bodypoint[rs][0])
#             elif not person_bodypoint[rs][0]:  # RShoulder == 0
#                 _w = min(person_bodypoint[ls][0], FRAME_W - person_bodypoint[ls][0])
#         else:  # Neck != 0 and MidHip != 0
#             _w = abs(person_bodypoint[ls][0] - person_bodypoint[rs][0])
#
#     _w = int(_w)
#     return _x, _y, _w, _h


# json파일 첫 frame에서 bbox 찾고 동영상에 tracking 적용하기
# with open('../output/output.json') as f:
#     json_data = json.load(f)
#
#     obj = json_data[0]['person'][0]['keypoint']  # 0번째 frame 0번째 person
#
#     tmp = []
#     for key in body_point:
#         if key == 'Background': break
#         tmp.append(obj[key])
#
#     dict_to_list = []
#     for dict in tmp:
#         tmp2 = []
#         for k, v in dict.items():
#             tmp2.append(v)
#         dict_to_list.append(tmp2)
#
#     VIDEO_PATH = '../output/openpose_output.avi'
#     cap = cv2.VideoCapture(VIDEO_PATH)
#     if cap.isOpened():
#         ret, frame = cap.read()
#         set_frame_size(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#         x, y, w, h = find_bbox2(dict_to_list)  # check_boundary(dict_to_list, FRAME_W, FRAME_H)
#         print(x, y, w, h)
#         obj_tracking(x, y, w, h)


# frame_data = detect_skeleton()
# get_location(frame_data)

print()  # 위쪽 주석이 아래 주석이랑 합쳐져서 임시로 넣어둠


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


# 한 사람의 모든 관절의 (x, y) 쌍을 리스트로 반환
def get_point_list(person):
    tmp = []
    for i in range(25):
        bt = person[body_point[i]]
        tmp.append((float(bt['x']), float(bt['y'])))
    return np.asarray(tmp)


# specific(추적할 객체)과 person(관측된 관절/사람)의 좌표 거리 차이
def person_distance(specific_bt, person_bt):
    dist = 0
    num_cnt = 0
    for i in range(25):
        if 0 in specific_bt[i] or 0 in person_bt[i]:
            continue

        dist += abs(specific_bt[i][0] - person_bt[i][0])
        dist += abs(specific_bt[i][1] - person_bt[i][1])
        num_cnt += 1

    if num_cnt is 0: return 0
    return dist / num_cnt

# 한 동영상에서-사람별-관절별-각도, 기울기 변화


# 특정 프레임수 단위로 끊어 그 값들을 평균을 낸다
def get_avg(val_list):
    num_avg = 10
    tmp = []
    s = len(val_list)
    iter = s // num_avg

    for i in range(iter):
        val = i*num_avg
        tmp.append(np.sum(val_list[val:val + num_avg]) / num_avg)

    if s % num_avg > 0:
        tmp.extend(val_list[num_avg * iter:])
    return tmp


# 0인 값 ok
# 너무 큰 값이 나왔을때(정확도가 낮은 경우) 0.5 ok?
# 사람별 변화값 분리 필요


# person id는 plotting 결과를 얻고자 함수를 반복하기 위해 임의로 넣은 것
def get_variance(json_filename, person_id, point_number):
    with open(json_filename) as f:
        json_data = json.load(f)
        json_len = len(json_data)
        angle_list = []
        incl_list = []

        num_pass = np.ones(4)  # 4개 팔다리에 대한 관절 측정 불가로 넘어간 프레임의 수
        pre_list = [[0.0, [0, 0]] for _ in range(4)]  # 4개의 팔다리에 대한 직전 각도와 (수학적)벡터값 저장
        specific = json_data[0]['person'][person_id]['keypoint']  # 우리가 원하는 특정한 객체 specific!
        specific_bt = get_point_list(specific)

        for i in range(1, json_len):
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
            dist_list = np.array(dist_list)
            specific_id = dist_list.argmin()
            specific = obj[specific_id]['keypoint']
            specific_bt = get_point_list(specific)

            # 추적하고자 하는 객체의 관절 변화를 계산
            p1, p2, p3 = specific_point[point_number]

            # 추적하는 객체의 관절값 가져오기
            p1 = specific[p1]; p2 = specific[p2]; p3 = specific[p3]
            acc1 = float(p1['accuracy']); acc2 = float(p2['accuracy']); acc3 = float(p3['accuracy'])
            p1 = (p1['x'], p1['y']); p2 = (p2['x'], p2['y']); p3 = (p3['x'], p3['y'])

            # 0이 있으면 해당 frame 건너 뛴다
            if 0 in p1 or 0 in p2 or 0 in p3 or \
                    acc1 < 0.5 or acc2 < 0.5 or acc3 < 0.5:
                num_pass[point_number] += 1
                continue

            # 세 점 사이의 각도를 구하여 직전 각도의 차이를 기록한다
            cur_angle = angle_three_points(p1, p2, p3)
            sub_angle = abs(pre_list[point_number][0] - cur_angle) / num_pass[point_number]
            if i > 1: angle_list.append(sub_angle)
            pre_list[point_number][0] = cur_angle

            # 두 점을 잇는 벡터와 직전 벡터와의 각도를 구한 후, 기록한다
            cur_vec = make_vector(p1, p3)
            pre_vec = pre_list[point_number][1]
            sub_incl_angle = get_incl_angle(cur_vec, pre_vec) / num_pass[point_number]
            if i > 1: incl_list.append(sub_incl_angle)
            pre_list[point_number][1] = cur_vec

            # 여기까지 왔으면 frame 안넘어갔겠다
            num_pass[point_number] = 1

    # n개의 frame을 단위로 그 값을 평균을 낸다 => 1초가 몇 frame인지 고려하면 좋을 듯
    avg_angle = get_avg(angle_list)
    avg_incl = get_avg(incl_list)
    return avg_angle, avg_incl


if __name__ == "__main__":
    # 값 변화 json으로 저장하기
    calc_variance()

    # 값 변화 plotting 하기
    non_violence_json_file = ['output47', 'output49']
    violence_json_file = ['output', 'output37']

    # plotting에 쓸 리스트
    angle_list = []
    incl_list = []
    # 폭력 데이터셋
    for name in violence_json_file:
        tmp_angle, tmp_incl = get_variance(name, 0)
        angle_list.extend(tmp_angle)
        incl_list.extend(tmp_incl)
        tmp_angle, tmp_incl = get_variance(name, 1)
        angle_list.extend(tmp_angle)
        incl_list.extend(tmp_incl)
    plt.scatter(incl_list, angle_list, label="violence")

    # 다른 색으로 적용하기 위해 리스트 다시 초기화
    angle_list = []
    incl_list = []
    # 비폭력 데이터셋
    for name in non_violence_json_file:
        tmp_angle, tmp_incl = get_variance(name, 0)
        angle_list.extend(tmp_angle)
        incl_list.extend(tmp_incl)
        tmp_angle, tmp_incl = get_variance(name, 1)
        angle_list.extend(tmp_angle)
        incl_list.extend(tmp_incl)
    plt.scatter(incl_list, angle_list, c='red', label="non-violence")

    plt.xlabel('inclination variance')
    plt.ylabel('angle variance')
    plt.legend()
    plt.show()

    # 일정 이상의 수치가 나오는 놈들 출력
    l = len(angle_list)
    for i in range(l):
        if angle_list[i] > 50 or incl_list[i] > 100:
            print(angle_list[i], incl_list[i])
