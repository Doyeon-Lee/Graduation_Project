from global_data import *
from main import *


def make_skeleton_json(datum, frame_id):
    if datum.poseKeypoints is None:
        return
    keypoints_tmp = []
    keypoints = {'frame_id': frame_id}
    for person_id in range(datum.poseKeypoints.shape[0]):

        location = {}
        for keypoint in range(15):
            body = {
                "x": float(datum.poseKeypoints[person_id][keypoint][0]),
                "y": float(datum.poseKeypoints[person_id][keypoint][1]),
                "accuracy": float(datum.poseKeypoints[person_id][keypoint][2])
            }
            location.update({body_point[keypoint]: body})

        keypoints_tmp.append({
            "person_id": person_id,
            "keypoint": location
        })
    keypoints["person"] = keypoints_tmp
    return keypoints


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
        point = {specific_joint[point_number]: variance}
        variance_list.append(point) # 4가지 모두 저장

    result["variance"] = variance_list
    return result


# skeleton json 에서 변화량 추출하기
def calc_variance():
    variance_data = []

    # 비폭력 데이터 cam1(다른 파일의 경우 global_data 파일의 경로를 수정해주자!)
    for i in range(1, 61):
        SKELETON_FILENAME = SKELETON_FILEPATH + str(i) + ".json"
        PREPROCESSED_FILENAME = PREPROCESSED_FILEPATH + str(i) + ".json"

        # 하나의 관절 json 파일에 대하여
        with open(SKELETON_FILENAME) as f:
            json_data = json.load(f)
            num_people = len(json_data[0]['person'])

            # 0번째 프레임에 있는 사람 수만큼 반복한다
            for person_id in range(num_people):
                variance_data.append(make_preprocessed_json(SKELETON_FILENAME, person_id))

            with open(PREPROCESSED_FILENAME, 'w', encoding="utf-8") as make_file:
                json.dump(variance_data, make_file, ensure_ascii=False, indent="\t")

