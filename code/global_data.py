BODY_POINT = ["Head", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
              "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "Chest", "Background"]


BODY_DICT = {}
for i, v in enumerate(BODY_POINT):
    BODY_DICT[v] = i

SPECIFIC_POINT = [
    ['LShoulder', 'LElbow', 'LWrist'],
    ['RShoulder', 'RElbow', 'RWrist'],
    ['LHip', 'LKnee', 'LAnkle'],
    ['RHip', 'RKnee', 'RAnkle']
]

SPECIFIC_JOINT = ['LArm', 'RArm', 'LLeg', 'RLeg']
NONV_SKELETON_FILEPATH = "../output/json/skeleton_data/non-violent/cam2/output"
V_SKELETON_FILEPATH = "../output/json/skeleton_data/violent/cam2/output"
NONV_PREPROCESSED_FILEPATH = "../output/json/preprocessed_data/non-violent/cam2/output"
V_PREPROCESSED_FILEPATH = "../output/json/preprocessed_data/violent/cam2/output"


# default video name
VIDEO_NAME = ""


def set_video_name(video_name):
    global VIDEO_NAME
    VIDEO_NAME = video_name


def get_video_name():
    return VIDEO_NAME


# ratio sum & people
RATIO_SUM = 0
PEOPLE_NUM = 0


def init_rate():
    global RATIO_SUM, PEOPLE_NUM
    RATIO_SUM = 0
    PEOPLE_NUM = 0


def set_rate(ratio_sum):
    global RATIO_SUM, PEOPLE_NUM
    RATIO_SUM += ratio_sum
    PEOPLE_NUM += 1


def get_rate():
    return RATIO_SUM, PEOPLE_NUM


# default frame size
FRAME_W = 0
FRAME_H = 0


def set_frame_size(w, h):
    global FRAME_W, FRAME_H
    # 32의 배수로 맞춰줌(FairMOT 조건)
    FRAME_W = w // 32 * 32
    FRAME_H = h // 32 * 32


def get_frame_size():
    return FRAME_W, FRAME_H


# 영상의 FRAME 개수
FRAME_NUM = 0


def set_frame_num(n):
    global FRAME_NUM
    FRAME_NUM = n


def get_frame_num():
    return FRAME_NUM


# 성인의 관절 리스트
SKELETON_LIST = []


def append_skeleton_list(value):
    global SKELETON_LIST
    SKELETON_LIST.append(value)


def extend_skeleton_list(value):
    global SKELETON_LIST
    SKELETON_LIST.extend(value)


def get_skeleton_list():
    return SKELETON_LIST


# 성인의 관절 좌표값
CURRENT_ADULT_POINT = {}
PREV_ADULT_POINT = {}


def init_point():
    global CURRENT_ADULT_POINT, PREV_ADULT_POINT
    for key in BODY_POINT:
        CURRENT_ADULT_POINT[key] = {"x": 0, "y": 0, "accuracy": 0}
        PREV_ADULT_POINT[key] = {"x": 0, "y": 0, "accuracy": 0}


def set_current_adult_point(points_dict):
    global CURRENT_ADULT_POINT
    CURRENT_ADULT_POINT = points_dict


def get_current_adult_point():
    return CURRENT_ADULT_POINT


def set_prev_adult_point(points_dict):
    global PREV_ADULT_POINT, PREV_ADULT_HEAD_LEN
    PREV_ADULT_POINT = points_dict

    # 성인의 머리 길이를 계산하여 저장
    Head = PREV_ADULT_POINT["Head"]
    if PREV_ADULT_POINT["RShoulder"]['accuracy'] < 0.7:
        Shoulder = PREV_ADULT_POINT["LShoulder"]
    elif PREV_ADULT_POINT["LShoulder"]['accuracy'] < 0.7:
        Shoulder = PREV_ADULT_POINT["RShoulder"]
    else:
        # 어깨의 가운데 점을 사용
        Shoulder = {'x': (PREV_ADULT_POINT["RShoulder"]['x'] + PREV_ADULT_POINT["LShoulder"]['x']) / 2, \
                    'y': (PREV_ADULT_POINT["RShoulder"]['y'] + PREV_ADULT_POINT["LShoulder"]['y']) / 2}
    PREV_ADULT_HEAD_LEN = ((Head['x'] - Shoulder['x']) ** 2 + (Head['y'] - Shoulder['y']) ** 2) ** 0.5


def get_prev_adult_point():
    return PREV_ADULT_POINT


# 성인의 머리 길이
PREV_ADULT_HEAD_LEN = 0


def get_prev_adult_head_len():
    return PREV_ADULT_HEAD_LEN


# 기타 함수 구현
def swap(x, y):
    return y, x


