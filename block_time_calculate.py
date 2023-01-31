#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file:统计车辆阻挡时间.py
@time:2023/01/28 20:25:03
@author:Yao Yongrui
'''

"""
1. 检测出现在五边形区域内的车辆，统计其阻挡时间
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import os
from ultralytics import YOLO

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="./fv_videos", help="name of input floder which contains videos")
args = parser.parse_args()


# 判断点是否在检测区域内
def is_in_region(point):
    # 六边形区域
    pts = np.array([[0, 288], [0, 256], [96, 144], [226, 144], [352, 256], [352, 288]], np.int32)
    # 判断点是否在区域内
    return cv2.pointPolygonTest(pts, point, False) >= 0


def draw_region(frame):
    # 六边形区域
    pts = np.array([[0, 288], [0, 256], [96, 144], [226, 144], [352, 256], [352, 288]], np.int32)
    zeros = np.zeros((288, 352, 3), np.int32)
    mask = cv2.fillPoly(zeros, [pts], color=(0, 0, 255))
    mask_img = cv2.addWeighted(frame, 1, mask, 0.5, 0, dtype=cv2.CV_8U)
    return mask_img


# 对单个视频检测阻挡时间
def detect_signal(video_path, model, output_floder):
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    # 获取视频帧数
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 输出视频，用于阻挡车辆检测可视化
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    paths_segmentation = video_path.split("/")
    video_name = paths_segmentation[-1]
    out = cv2.VideoWriter(f"{output_floder}/{video_name[:-4]}_detect.avi", fourcc, 12.5, (352, 288))
    
    # 统计阻挡帧数
    block_frame_num = 0

    for i in range(total_frame):
        # 读取视频帧
        ret, frame = cap.read()
        print(f"\r{i+1}/{total_frame}", end="")
        if ret:
            results = model.predict(frame)
            result = results[0]
            # 获取检测类别
            cls_orginal = result.boxes.cls
            # 获取置信度
            conf_orginal = result.boxes.conf
            # 获取检测框位置
            boxes_orginal = result.boxes.xywh
            # 用于绘制检测框
            boxes_draw = result.boxes.xyxy

            # 筛选检测结果
            cls_filter = []
            conf_filter = []
            # 检测点位置
            dpoint_filter = []
            # 与coco数据集中的标号一一对应
            # 2-car
            # 5-bus
            # 7-truck
            cls_detect = [2, 5, 7]

            # 标示一帧中是否有阻挡车辆
            block_frame_flag = 0

            # 绘制检测区域，用于可视化
            frame = draw_region(frame)

            for i in range(len(cls_orginal)):
                # 筛选出车辆
                if cls_orginal[i] in cls_detect and conf_orginal[i] > 0.3:
                    # xc和yc为检测框中心点坐标
                    xc, yc, _, h = boxes_orginal[i]
                    xmin, ymin, xmax, ymax = boxes_draw[i]
                    # 检测点为检测框宽度的中心和高度的2/3处
                    dpoint_x = int(xc)
                    dpoint_y = int(yc + h / 6)
                    dpoint_tuple = (dpoint_x, dpoint_y)
                    # 判断中心点是否在检测区域内
                    if is_in_region(dpoint_tuple):
                        # 绘制检测框
                        frame = cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 109, 237), 2)
                        # 为了美观可以在此处添加标签和置信度等文字
                        block_frame_flag = 1
            block_frame_num += block_frame_flag

            out.write(frame)

        else:
            print("视频读取失败")
    
    # 输出阻挡时间
    block_second_num = block_frame_num / 12.5
    print('\n')
    cap.release()
    out.release()

    return block_second_num


if __name__ == '__main__':
    # 加载模型
    model = YOLO("./model_file/yolov8s.pt")
    # 获取视频文件夹
    videos = os.listdir(args.input)
    # 统计文件夹内所有视频的阻挡时间
    block_time_ls = []

    # 对每个视频进行检测
    for video in videos:
        print(f"{video}处理进度：")
        block_second_num = detect_signal(f"{args.input}/{video}", model, './block_detect_output')
        block_time_ls.append(block_second_num)
    
    # 输出阻挡时间至excel
    pd.DataFrame({'阻挡时间/s': block_time_ls}).to_excel('./block_time.xlsx', index=False)
    