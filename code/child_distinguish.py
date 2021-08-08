import json
import numpy as np


def child_distinguish(file_name, frame_num):
    # json 파일 열기
    with open(f'../output/json/{file_name}.json', 'r') as f:
        json_data = json.load(f)

    body_ratio_list = np.array([])
    average = 0
    for person_num in range(0, len(json_data[frame_num]['person'])):
        Head = json_data[frame_num]['person'][person_num]['keypoint']['Head']
        Neck = json_data[frame_num]['person'][person_num]['keypoint']['Neck']

        RHip = json_data[frame_num]['person'][person_num]['keypoint']['RHip']
        LHip = json_data[frame_num]['person'][person_num]['keypoint']['LHip']

        if RHip['accuracy'] < 0.5:
            len_body = ((Neck['x'] - LHip['x']) ** 2 + (Neck['y'] - LHip['y']) ** 2) ** 0.5
        elif LHip['accuracy'] < 0.5:
            len_body = ((Neck['x'] - RHip['x']) ** 2 + (Neck['y'] - RHip['y']) ** 2) ** 0.5
        else:
            # 엉덩이의 가운데 점을 사용
            Hip = {'x': (RHip['x'] + LHip['x']) / 2, 'y': (RHip['y'] + LHip['y']) / 2}

            len_body = ((Neck['x'] - Hip['x'])**2 + (Neck['y'] - Hip['y'])**2)**0.5

        len_head = ((Head['x'] - Neck['x'])**2 + (Head['y'] - Neck['y'])**2)**0.5

        body_ratio = len_head / (len_head + len_body)

        body_ratio_list = np.append(body_ratio_list, body_ratio)
        average += body_ratio

    candidate_ratio = min(body_ratio_list)
    candidate_key = np.where(body_ratio_list == candidate_ratio)[0][0]

    # 후보를 제외한 평균과 비교했을 때 차이가 0.03 이상 나면 어른으로 간주
    average = (average - candidate_ratio) / (len(json_data[frame_num]['person']) - 1)
    if average - candidate_ratio >= 0.03:
        return candidate_key

    # 성인이 없다면 -1을 return
    return -1

