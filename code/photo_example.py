# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import json
import numpy as np

# 사용자가 보는 것과 반대 방향(오른쪽 > left)
body_point = ["Head", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
              "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "Chest", "Background"]


def make_json(datum, frame_id, img):
    keypoints_tmp = []
    keypoints = {'frame_id': frame_id}

    for person_id in range(datum.poseKeypoints.shape[0]):
        person_bodypoint = datum.poseKeypoints[person_id]

        tmp = int(person_bodypoint[5][0] - person_bodypoint[2][0])
        # img = cv2.rectangle(img, (int(person_bodypoint[5][0]), int(person_bodypoint[5][1] - tmp)), (\
        #     int(person_bodypoint[2][0]), int(person_bodypoint[22][1])), (0, 255, 0), 3)
        img = cv2.putText(img, str(person_id), (int(person_bodypoint[3][0]), int(person_bodypoint[5][1] - tmp)), \
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 4)

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

    cv2.imshow('test', img)
    cv2.waitKey(0)
    return keypoints



try:
    # Import Openpose (Windows/Ubuntu/OSX)
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../openpose/build/python/openpose/Release');
            os.environ['PATH'] = os.environ['PATH'] + ';' + '../../openpose/build/x64/Release;' + '../../openpose/build/bin'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../openpose/build/python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there.
            # This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../output/video/54/frame/00000.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../openpose/models/"
    params["model_pose"] = "MPI"
    params["net_resolution"] = "-1x160"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # Display Image
    print("Body keypoints: \n" + str(datum.poseKeypoints))
    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    cv2.waitKey(0)

    # save the keypoint as a list
    draw_img = cv2.imread("../output/video/54/frame/00000.jpg", cv2.IMREAD_COLOR)
    picture_json = [make_json(datum, 0, draw_img)]

    # with open('../output/json/picture/family1.json', 'w', encoding="utf-8") as make_file:
    #     json.dump(picture_json, make_file, ensure_ascii=False, indent="\t")


except Exception as e:
    print(e)
    sys.exit(-1)
