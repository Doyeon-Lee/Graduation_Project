import json


def child_distinguish(file_name):
    # json 파일 열기
    with open(f'../output/json/{file_name}.json', 'r') as f:
        json_data = json.load(f)

    for frame_num in range(0, len(json_data)):
        body_ratio_dict = {}
        average = 0
        for person_num in range(0, len(json_data[frame_num]['person'])):
            Neck = json_data[frame_num]['person'][person_num]['keypoint']['Neck']
            MidHip = json_data[frame_num]['person'][person_num]['keypoint']['MidHip']

            REar = json_data[frame_num]['person'][person_num]['keypoint']['REar']
            LEar = json_data[frame_num]['person'][person_num]['keypoint']['LEar']

            if REar['accuracy'] < 0.5:
                len_head = ((LEar['x'] - Neck['x']) ** 2 + (LEar['y'] - Neck['y']) ** 2) ** 0.5
            elif LEar['accuracy'] < 0.5:
                len_head = ((REar['x'] - Neck['x']) ** 2 + (REar['y'] - Neck['y']) ** 2) ** 0.5
            else:
                # 귀의 가운데 점을 사용
                Ear = {'x': (REar['x'] + LEar['x']) / 2, 'y': (REar['y'] + LEar['y']) / 2}

                len_head = ((Ear['x'] - Neck['x'])**2 + (Ear['y'] - Neck['y'])**2)**0.5

            len_body = ((Neck['x'] - MidHip['x'])**2 + (Neck['y'] - MidHip['y'])**2)**0.5

            body_ratio = len_head / (len_head + len_body)

            body_ratio_dict[person_num] = body_ratio
            average += body_ratio

        candidate_key = min(body_ratio_dict)
        candidate_ratio = min(body_ratio_dict.values())

        print(body_ratio_dict)

        # 후보를 제외한 평균과 비교했을 때 차이가 0.03 이상 나면 어른으로 간주
        average = (average - candidate_ratio) / (len(json_data[frame_num]['person']) - 1)
        if average - candidate_ratio >= 0.03:
            return candidate_key


print(child_distinguish('picture/family1'))
print(child_distinguish('picture/family2'))
print(child_distinguish('picture/family3'))
