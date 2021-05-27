import numpy as np
import sys
import cv2
import os
from sys import platform
import argparse
import json
from math import atan2, degrees
import matplotlib.pyplot as plt


body_point = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist",
              "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
              "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
              "REye", "LEye", "REar", "LEar", "LBigToe",
              "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel", "Background"]

body_dict = {}
for i, v in enumerate(body_point):
    body_dict[v] = i

# default frame size
FRAME_W = 0
FRAME_H = 0


def set_frame_size(w, h):
    global FRAME_W, FRAME_H
    FRAME_W = w
    FRAME_H = h


def get_frame_size():
    return FRAME_W, FRAME_H

