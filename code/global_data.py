body_point = ["Head", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
              "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "Chest", "Background"]


body_dict = {}
for i, v in enumerate(body_point):
    body_dict[v] = i

specific_point = [
    ['LShoulder', 'LElbow', 'LWrist'],
    ['RShoulder', 'RElbow', 'RWrist'],
    ['LHip', 'LKnee', 'LAnkle'],
    ['RHip', 'RKnee', 'RAnkle']
]

specific_joint = ['LArm', 'RArm', 'LLeg', 'RLeg']
NONV_SKELETON_FILEPATH = "../output/json/skeleton_data/non-violent/cam2/output"
V_SKELETON_FILEPATH = "../output/json/skeleton_data/violent/cam2/output"
NONV_PREPROCESSED_FILEPATH = "../output/json/preprocessed_data/non-violent/cam2/output"
V_PREPROCESSED_FILEPATH = "../output/json/preprocessed_data/violent/cam2/output"

# default frame size
FRAME_W = 0
FRAME_H = 0

# default video name
VIDEO_NAME = ""

# ratio sum & people
RATIO_SUM = 0
PEOPLE = 0

# 성인의 관절 좌표값
CURRENT_POINT_OF_ADULT = {
    "Head": {"x": 0, "y": 0, "accuracy": 0},
    "Neck": {"x": 0, "y": 0, "accuracy": 0},
    "RShoulder": {"x": 0, "y": 0, "accuracy": 0},
    "RElbow": {"x": 0, "y": 0, "accuracy": 0},
    "RWrist": {"x": 0, "y": 0, "accuracy": 0},
    "LShoulder": {"x": 0, "y": 0, "accuracy": 0},
    "LElbow": {"x": 0, "y": 0, "accuracy": 0},
    "LWrist": {"x": 0, "y": 0, "accuracy": 0},
    "RHip": {"x": 0, "y": 0, "accuracy": 0},
    "RKnee": {"x": 0, "y": 0, "accuracy": 0},
    "RAnkle": {"x": 0, "y": 0, "accuracy": 0},
    "LHip": {"x": 0, "y": 0, "accuracy": 0},
    "LKnee": {"x": 0, "y": 0, "accuracy": 0},
    "LAnkle": {"x": 0, "y": 0, "accuracy": 0},
    "Chest": {"x": 0, "y": 0, "accuracy": 0}
}
PREV_POINT_OF_ADULT = {
    "Head": {"x": 0, "y": 0, "accuracy": 0},
    "Neck": {"x": 0, "y": 0, "accuracy": 0},
    "RShoulder": {"x": 0, "y": 0, "accuracy": 0},
    "RElbow": {"x": 0, "y": 0, "accuracy": 0},
    "RWrist": {"x": 0, "y": 0, "accuracy": 0},
    "LShoulder": {"x": 0, "y": 0, "accuracy": 0},
    "LElbow": {"x": 0, "y": 0, "accuracy": 0},
    "LWrist": {"x": 0, "y": 0, "accuracy": 0},
    "RHip": {"x": 0, "y": 0, "accuracy": 0},
    "RKnee": {"x": 0, "y": 0, "accuracy": 0},
    "RAnkle": {"x": 0, "y": 0, "accuracy": 0},
    "LHip": {"x": 0, "y": 0, "accuracy": 0},
    "LKnee": {"x": 0, "y": 0, "accuracy": 0},
    "LAnkle": {"x": 0, "y": 0, "accuracy": 0},
    "Chest": {"x": 0, "y": 0, "accuracy": 0}
}


def set_frame_size(w, h):
    global FRAME_W, FRAME_H
    # 32의 배수로 맞춰줌(FairMOT 조건)
    FRAME_W = w // 32 * 32
    FRAME_H = h // 32 * 32


def get_frame_size():
    return FRAME_W, FRAME_H


def get_video_name():
    return VIDEO_NAME


def set_video_name(video_name):
    global VIDEO_NAME
    VIDEO_NAME = video_name


def init_rate():
    global RATIO_SUM, PEOPLE
    RATIO_SUM = 0
    PEOPLE = 0


def set_rate(ratio_sum):
    global RATIO_SUM, PEOPLE
    RATIO_SUM += ratio_sum
    PEOPLE += 1


def get_rate():
    return RATIO_SUM, PEOPLE


def set_current_adult_point(points_dict):
    global CURRENT_POINT_OF_ADULT
    CURRENT_POINT_OF_ADULT = points_dict


def get_current_adult_point():
    return CURRENT_POINT_OF_ADULT


def set_prev_adult_point(points_dict):
    global PREV_POINT_OF_ADULT
    PREV_POINT_OF_ADULT = points_dict


def get_prev_adult_point():
    return PREV_POINT_OF_ADULT
