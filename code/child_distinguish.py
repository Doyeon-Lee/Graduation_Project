import json
import numpy as np
from global_data import set_rate, get_rate


def child_distinguish(file_name, frame_num):
    # json 파일 열기
    with open(file_name, 'r') as f:
        json_data = json.load(f)

    body_ratio_list = np.array([])
    for person_num in range(0, len(json_data[frame_num]['person'])):
        Head = json_data[frame_num]['person'][person_num]['keypoint']['Head']
        Chest = json_data[frame_num]['person'][person_num]['keypoint']['Chest']

        RShoulder = json_data[frame_num]['person'][person_num]['keypoint']['RShoulder']
        LShoulder = json_data[frame_num]['person'][person_num]['keypoint']['LShoulder']

        RHip = json_data[frame_num]['person'][person_num]['keypoint']['RHip']
        LHip = json_data[frame_num]['person'][person_num]['keypoint']['LHip']

        # 머리/어깨/가슴/엉덩이가 인식이 안되면 넘어감
        if Head['accuracy'] < 0.5 or (RShoulder['accuracy'] < 0.5 and LShoulder['accuracy'] < 0.5) or\
            Chest['accuracy'] < 0.5 or (RHip['accuracy'] < 0.5 and LHip['accuracy'] < 0.5):
            continue

        if RShoulder['accuracy'] < 0.5:
            Shoulder = LShoulder
        elif LShoulder['accuracy'] < 0.5:
            Shoulder = RShoulder
        else:
            # 어깨의 가운데 점을 사용
            Shoulder = {'x': (RShoulder['x'] + LShoulder['x']) / 2, 'y': (RShoulder['y'] + LShoulder['y']) / 2}

        len_head = ((Head['x'] - Shoulder['x']) ** 2 + (Head['y'] - Shoulder['y']) ** 2) ** 0.5

        if RHip['accuracy'] < 0.5:
            Hip = LHip
        elif LHip['accuracy'] < 0.5:
            Hip = RHip
        else:
            # 엉덩이의 가운데 점을 사용
            Hip = {'x': (RHip['x'] + LHip['x']) / 2, 'y': (RHip['y'] + LHip['y']) / 2}

        len_body = ((Shoulder['x'] - Chest['x'])**2 + (Shoulder['y'] - Chest['y'])**2)**0.5 + \
                   ((Chest['x'] - Hip['x'])**2 + (Chest['y'] - Hip['y'])**2)**0.5

        if len_head == 0 or len_head + len_body == 0:
            continue
        else:
            body_ratio = len_head / (len_head + len_body)

            body_ratio_list = np.append(body_ratio_list, body_ratio)
            set_rate(body_ratio)

    if len(body_ratio_list) == 0:
        return -1

    candidate_ratio = min(body_ratio_list)
    candidate_key = np.where(body_ratio_list == candidate_ratio)[0][0]

    ratio_sum, people = get_rate()
    average = ratio_sum / people

    print(average, candidate_key, body_ratio_list)
    # 표준편차값이 0.0175보다 작으면 성인이 감지되지 않았다고 생각
    # 성인의 자세가 흐트러졌을 경우 관절을 잘못 인식할 수 있기 때문에 제외
    if np.std(body_ratio_list) < 0.0175:
        return -1
    else:
        # 성인이 많이 인식되어 평균값이 성인에 가까울 때
        if average < 0.45:
            if abs(average - candidate_ratio) <= 0.024:
                return candidate_key
        # 아이가 많이 인식되어 평균값이 아이에 가까울 때
        else:
            if average - candidate_ratio >= 0.024:
                return candidate_key

    # 성인이 없다면 -1을 return
    return -1
