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
