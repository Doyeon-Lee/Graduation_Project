import numpy as np
import sys
import cv2
import os
from sys import platform
import argparse
import json
from math import atan2, degrees
import matplotlib.pyplot as plt


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
NONV_SKELETON_FILEPATH = "../output/json/skeleton_data/non-violent/cam1/output"
V_SKELETON_FILEPATH = "../output/json/skeleton_data/violent/cam2/output"
NONV_PREPROCESSED_FILEPATH = "../output/json/preprocessed_data/non-violent/cam1/output"
V_PREPROCESSED_FILEPATH = "../output/json/preprocessed_data/violent/cam2/output"

# default frame size
FRAME_W = 0
FRAME_H = 0


def set_frame_size(w, h):
    global FRAME_W, FRAME_H
    FRAME_W = w
    FRAME_H = h


def get_frame_size():
    return FRAME_W, FRAME_H

