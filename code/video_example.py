# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import json

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
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="../media/17.mp4", help="Read input video (avi, mp4).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../openpose/models/"
    params["disable_multi_thread"] = "false"
    numberGPUs = 1

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

    # Process Video
    datum = op.Datum()
    cap = cv2.VideoCapture(args[0].video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    # out = cv2.VideoWriter('../output/output.avi', fourcc, 20.0, (width, height))

    body_point = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
                  "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
                  "REye", "LEye", "REar", "LEar",
                  "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel", "Background"]

    frame_data = []
    frame_id = -1
    while cap.isOpened():
        frame_id += 1

        grabbed, frame = cap.read()

        if frame is None or not grabbed:
            print("Finish reading video frames...")
            break

        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        # print("Body keypoints: \n" + str(datum.poseKeypoints))

        # # write the flipped frame
        # out.write(datum.cvOutputData)

        # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
        # cv2.waitKey(1)

        # save the keypoint as a list
        keypoints_tmp = []
        keypoints = {'frame_id': frame_id}
        for person_id in range(datum.poseKeypoints.shape[0]):
            location = {}
            for keypoint in range(25):
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
        frame_data.append(keypoints)

    else:
        print('cannot open the file')

    # show list as json
    with open('../output/output.json', 'w', encoding="utf-8") as make_file:
        json.dump(frame_data, make_file, ensure_ascii=False, indent="\t")

    cap.release()
    # out.release()
    # cv2.destroyAllWindows()
except Exception as e:
    print(e)
    sys.exit(-1)
