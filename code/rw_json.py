from global_data import *


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
