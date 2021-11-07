
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from plotting import get_variance
import os


# # 1) Model
# # Linear model f = wx + b , sigmoid at the end
# class Model(nn.Module):
#     def __init__(self, n_input_features):
#         super(Model, self).__init__()
#         self.linear = nn.Linear(n_input_features, 1)
#
#     def forward(self, x):
#         y_pred = torch.sigmoid(self.linear(x))
#         return y_pred
#
#
# def setData(dataset):
#     # 0) Prepare data
#     JSON_PATH = "../output/json/final_data"
#     file_lst = os.listdir(JSON_PATH)
#
#     for file_name in file_lst:
#         file_path = JSON_PATH + f'{file_name}'
#         print(file_path)
#         for i in range(4):
#             angle, incli = get_variance(file_name, i)
#             angle = np.array(angle).reshape(-1, 1)
#             incli = np.array(incli).reshape(-1, 1)
#             tmp = np.concatenate((angle, incli), axis=1)
#
#             if i < 2:
#                 if len(dataset) == 0:
#                     dataset = tmp
#                 else:
#                     dataset = np.concatenate((dataset, tmp), axis=1)
#             else:
#                 dataset = np.concatenate((dataset, tmp), axis=1)
#
#
#
#         X_train = dataset
#         X_train = np.array(X_train).reshape(-1, 2)
#         X_test = [45, 44, 43, 41, 1, 2, 13, 9]
#         X_test = np.array(X_test).reshape(-1, 2)
#         y_train = [0, 0, 0, 0,
#                    1, 1, 1,
#                    1, 1]
#         y_train = np.array(y_train).reshape(-1, 1)
#         y_test = [1, 1, 0, 0]
#         y_test = np.array(y_test).reshape(-1, 1)
#
#         n_samples, n_features = X_train.shape
#
#         # scale
#         sc = StandardScaler()
#         X_train = sc.fit_transform(X_train)
#         X_test = sc.transform(X_test)
#
#         X_train = torch.from_numpy(X_train.astype(np.float32))
#         X_test = torch.from_numpy(X_test.astype(np.float32))
#         y_train = torch.from_numpy(y_train.astype(np.float32))
#         y_test = torch.from_numpy(y_test.astype(np.float32))
#
#         y_train = y_train.view(y_train.shape[0], 1)
#         y_test = y_test.view(y_test.shape[0], 1)
#
#
#
#
#
# # 2) Loss and optimizer
# num_epochs = 50
# learning_rate = 0.01
# criterion = nn.BCELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
# # 3) Training loop
# for epoch in range(num_epochs):
#     # Forward pass and loss
#     y_pred = model(X_train)
#     loss = criterion(y_pred, y_train)
#
#     # Backward pass and update
#     loss.backward()
#     optimizer.step()
#
#     # zero grad before new step
#     optimizer.zero_grad()
#
#     if (epoch+1) % 10 == 0:
#         print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
#
#
# with torch.no_grad():
#     y_predicted = model(X_test)
#     y_predicted_cls = y_predicted.round()
#     acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
#     print(f'accuracy: {acc.item():.4f}')
#
# def init_parameter():
#     num_epochs = 50
#     learning_rate = 0.01
#     criterion = nn.BCELoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#     model = Model(n_features)
#
#     dataset = []
#     setData(dataset)
#
# def train_model():


# if __name__ == "__main__":

import re
import cv2
from global_data import set_frame_size

dataset = []
JSON_PATH = "../output/json/final_data"
# file_lst = os.listdir(JSON_PATH)
file_lst = ["results600.json"]

cnt = 0
for file_name in file_lst:
    if cnt == 10:
        break

    file_path = JSON_PATH + "/" + file_name
    file_name = re.sub('.json', '', file_name)
    file_name = re.sub('results', '', file_name)
    VIDEO_PATH = f'../media/non-violence/{file_name}.mp4'
    print(VIDEO_PATH)

    # frame_size 정해줌
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    set_frame_size(int(width), int(height))
    cap.release()

    for i in range(4):
        angle, incli = get_variance(file_path, i)
        angle = np.array(angle).reshape(-1, 1)
        incli = np.array(incli).reshape(-1, 1)
        print(angle.shape, incli.shape)

        tmp = np.concatenate((angle, incli), axis=1)


        # if i < 2:
        #     if len(dataset) == 0:
        #         dataset = tmp
        #     else:
        #         dataset = np.concatenate((dataset, tmp), axis=1)
        # else:
        #     dataset = np.concatenate((dataset, tmp), axis=1)
    cnt += 1


