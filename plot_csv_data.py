import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_control_chart(data):
    # 判断数据是不是属于连续三点中有两点落在A区及A区之外
    R_list = [np.max(i) - np.min(i) for i in data]
    # 计算样本计算这25个样本点的CL、R_bar、UCL和LCL,σ
    X_bar_list = [np.mean(i) for i in data]
    CL = np.mean(X_bar_list)
    R_bar = np.mean(R_list)
    UCL = CL + 0.577 * R_bar
    LCL = CL - 0.577 * R_bar
    xigama = 1.154 * R_bar / 6
    plt.scatter(range(len(X_bar_list)), X_bar_list)
    plt.plot(range(len(X_bar_list)), X_bar_list)
    # 画出辅助线
    plt.plot(range(len(X_bar_list)), [CL] * 25, 'r')
    plt.plot(range(len(X_bar_list)), [UCL] * 25, 'r')
    plt.plot(range(len(X_bar_list)), [LCL] * 25, 'r')
    plt.plot(range(len(X_bar_list)), [CL + xigama] * 25, 'black')
    plt.plot(range(len(X_bar_list)), [CL - xigama] * 25, 'black')
    plt.plot(range(len(X_bar_list)), [CL + 2 * xigama] * 25, 'g')
    plt.plot(range(len(X_bar_list)), [CL - 2 * xigama] * 25, 'g')
    plt.show()
    plt.close()


def judge_3_type(data):
    # 判断是不是数据是不是属于第三类。即属于连续6点递增或者递减
    R_list = [np.max(i) - np.min(i) for i in data]
    # 计算样本计算这25个样本点的CL、R_bar、UCL和LCL
    X_bar_list = [np.mean(i) for i in data]
    CL = np.mean(X_bar_list)
    R_bar = np.mean(R_list)
    UCL = CL + 0.577 * R_bar
    LCL = CL - 0.577 * R_bar
    # 对这段数据求导数，然后去判断这些样本点中是不是有连续六点递增或递减
    flag = 0
    grid_X_bar_list = [X_bar_list[i + 1] - X_bar_list[i] for i in range(len(X_bar_list) - 1)]
    for i in range(len(grid_X_bar_list) - 6):
        # 截取六个点
        part_grid_X_bar_list = grid_X_bar_list[i:i + 6]
        # 如果都大于0了，就退出循环
        if (np.array(part_grid_X_bar_list) > 0).all():
            flag = 1
            break
        # 如果都小于0了，就退出循环
        if (np.array(part_grid_X_bar_list) < 0).all():
            flag = 1
            break
    return flag


def judge_4_type(data):
    # 判断是不是数据是不是属于是不是有连续14个点上下交替
    R_list = [np.max(i) - np.min(i) for i in data]
    # 计算样本计算这25个样本点的CL、R_bar、UCL和LCL
    X_bar_list = [np.mean(i) for i in data]
    CL = np.mean(X_bar_list)
    R_bar = np.mean(R_list)
    UCL = CL + 0.577 * R_bar
    LCL = CL - 0.577 * R_bar
    # 对这段数据求导数，然后去判断这些样本点中是不是有连续14个点上下交替
    flag = 0
    grid_X_bar_list = [X_bar_list[i + 1] - X_bar_list[i] for i in range(len(X_bar_list) - 1)]
    # 14个点上下交替的模板1
    template = [True, False, True, False, True, False, True, False, True, False, True, False, True, False]

    for i in range(len(grid_X_bar_list) - 14):
        # 截取14个点
        part_grid_X_bar_list = grid_X_bar_list[i:i + 14]
        # 看导数的第一个点，如果第一个点是大于0的，那应该匹配模板，如果匹配上了，就退出循环
        if list(np.array(part_grid_X_bar_list) > 0) == template:
            flag = 1
            break
        # 看导数的第一个点，如果第一个点是小于0的，那应该匹配模板，如果匹配上了，就退出循环
        if list(np.array(part_grid_X_bar_list) < 0) == template:
            flag = 1
            break
    return flag


def judge_5_type(data):
    # 判断数据是不是属于连续三点中有两点落在A区及A区之外
    R_list = [np.max(i) - np.min(i) for i in data]
    # 计算样本计算这25个样本点的CL、R_bar、UCL和LCL,σ
    X_bar_list = [np.mean(i) for i in data]
    CL = np.mean(X_bar_list)
    R_bar = np.mean(R_list)
    UCL = CL + 0.577 * R_bar
    LCL = CL - 0.577 * R_bar
    xigama = 1.154 * R_bar / 6
    # 对这段数据求导数，然后去判断这些样本点中是不是连续三点中有两点落在A区及A区之外
    flag = 0
    for i in range(len(X_bar_list) - 3):
        # 截取长度为三的X_bar
        part_X_bar_list = X_bar_list[i:i + 3]
        count_5_num = 0
        for X_bar in part_X_bar_list:
            if np.absolute(X_bar - CL) > (2 * xigama):
                count_5_num = count_5_num + 1
        if count_5_num > 2:
            flag = 1
            break
    return flag


if __name__ == '__main__':
    NEED_CHECK_TYPE_PATH = r'.\csv_data\连续三点中有两点落在A区及A区之外'
    for i in os.listdir(NEED_CHECK_TYPE_PATH):
        csv_path = os.path.join(NEED_CHECK_TYPE_PATH,i)
        batch_data = pd.read_csv(csv_path).values[:, 1:]
        plot_control_chart(batch_data)
