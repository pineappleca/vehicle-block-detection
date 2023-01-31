#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file:绘制检测区域.py
@time:2023/01/26 10:38:07
@author:Yao Yongrui
'''

"""
1. 在原始公交前方视角视频中绘制车辆阻挡检测区域
"""


import cv2
import numpy as np
import argparse
import os 


# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="test.avi", help="name of input video or floder")
args = parser.parse_args()


# 单个视频绘制函数
def draw_region_single(video_name):
    # 检测区域范围
    # 五边形区域
    # pts = np.array([[0, 288], [0, 265], [143, 148], [213, 148], [285, 288]], np.int32)
    # 四边形区域
    # pts = np.array([[35, 288], [137, 150], [215, 150], [317, 288]], np.int32)
    # 六边形区域
    pts = np.array([[0, 288], [0, 256], [96, 144], [156, 144], [352, 256], [352, 288]], np.int32)
    # 通过蒙版与原图加和的方式调整透明度
    zeros = np.zeros((288, 352, 3), np.int32)
    mask = cv2.fillPoly(zeros, [pts], color=(0, 0, 255))
    # 读取视频
    cap = cv2.VideoCapture(video_name)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(f'{video_name[:-4]}_region.avi', fourcc, 12.5, (352, 288))
    # 视频总帧数
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(total_frame):
        # 读取视频帧
        ret, frame = cap.read()
        print(f"\r{i+1}/{total_frame}", end="")
        if ret:
            mask_img = cv2.addWeighted(frame, 1, mask, 0.8, 0, dtype=cv2.CV_8U)
            out.write(mask_img)
        else:
            print("视频读取失败")
    cap.release()
    out.release()


# 多个视频绘制函数
def draw_region_multi(floder_name):
    # 检测区域范围
    # 五边形区域
    # pts = np.array([[0, 288], [0, 265], [143, 148], [213, 148], [285, 288]], np.int32)
    # 四边形区域
    # pts = np.array([[35, 288], [137, 150], [215, 150], [317, 288]], np.int32)
    # 六边形区域
    pts = np.array([[0, 288], [0, 256], [96, 144], [156, 144], [352, 256], [352, 288]], np.int32)
    # 通过蒙版与原图加和的方式调整透明度
    zeros = np.zeros((288, 352, 3), np.int32)
    mask = cv2.fillPoly(zeros, [pts], color=(0, 0, 255))

    # 获取文件夹下所有视频
    video_name_list = os.listdir(floder_name)
    for video_name in video_name_list:
        # 读取视频
        cap = cv2.VideoCapture(floder_name + "/" + video_name)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter("./region_output/" + video_name[:-4] + "_region.avi", fourcc, 12.5, (352, 288))
        # 视频总帧数
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{video_name}处理进度：")
        for i in range(total_frame):
            # 读取视频帧
            ret, frame = cap.read()
            print(f"\r{i+1}/{total_frame}", end="")
            if ret:
                mask_img = cv2.addWeighted(frame, 1, mask, 0.8, 0, dtype=cv2.CV_8U)
                out.write(mask_img)
            else:
                print("视频读取失败")
        print('\n')
        cap.release()
        out.release()


if __name__ == "__main__":
    draw_region_single(args.input)

