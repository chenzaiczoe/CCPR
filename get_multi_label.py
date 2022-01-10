import numpy as np


def judge_under_9_cl(X_bar_list, CL):
    flag = 1
    # 判断这么九个点是不是都在CL线下面
    for j in X_bar_list:
        # 9个点都在cl下方，退出循环
        if j > CL:
            flag = 0
            break
    return flag


def judge_up_9_cl(X_bar_list, CL):
    flag = 1
    # 判断这么九个点是不是都在CL线上面
    for j in X_bar_list:
        # 9个点都在cl下方，退出循环
        if j < CL:
            flag = 0
            break
    return flag


def judge_2_type(data):
    # 判断是不是数据是不是属于9个点在中心线同一侧
    R_list = [np.max(i) - np.min(i) for i in data]
    # 计算样本计算这25个样本点的CL、R_bar、UCL和LCL
    X_bar_list = [np.mean(i) for i in data]
    CL = np.mean(X_bar_list)
    R_bar = np.mean(R_list)
    UCL = CL + 0.577 * R_bar
    LCL = CL - 0.577 * R_bar
    # 对这段数据求导数，然后去判断这些样本点中是不是有连续14个点上下交替
    flag = 0
    for i in range(len(data) - 9):
        part_X_bar_list = X_bar_list[int(i): int(i + 9)]
        if judge_under_9_cl(part_X_bar_list, CL) or judge_up_9_cl(part_X_bar_list, CL):
            flag = 1
            break

    return flag


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


def judge_6_type(data):
    # 判断数据是不是属于连续五点中有四点落在B区域及B区域之外
    R_list = [np.max(i) - np.min(i) for i in data]
    # 计算样本计算这25个样本点的CL、R_bar、UCL和LCL,σ
    X_bar_list = [np.mean(i) for i in data]
    CL = np.mean(X_bar_list)
    R_bar = np.mean(R_list)
    UCL = CL + 0.577 * R_bar
    LCL = CL - 0.577 * R_bar
    xigama = 1.154 * R_bar / 6
    # 对这段数据求导数，然后去判断这些样本点中是不是连续五点中有四点落在B区域及B区域之外
    flag = 0
    for i in range(len(X_bar_list) - 5):
        # 截取长度为三的X_bar
        part_X_bar_list = X_bar_list[i:i + 5]
        count_6_num = 0
        for X_bar in part_X_bar_list:
            if np.absolute(X_bar - CL) > (xigama):
                count_6_num = count_6_num + 1
        if count_6_num > 4:
            flag = 1
            break
    return flag


def judge_7_type(data):
    # 判断数据是不是属于连续15个点都落在中心线上线的C区之内
    R_list = [np.max(i) - np.min(i) for i in data]
    # 计算样本计算这25个样本点的CL、R_bar、UCL和LCL,σ
    X_bar_list = [np.mean(i) for i in data]
    CL = np.mean(X_bar_list)
    R_bar = np.mean(R_list)
    UCL = CL + 0.577 * R_bar
    LCL = CL - 0.577 * R_bar
    xigama = 1.154 * R_bar / 6
    # 对这段数据求导数，然后去判断这些样本点中是不是连续15个点都落在中心线上线的C区之内
    flag = 0
    for i in range(len(X_bar_list) - 15):
        # 截取长度为三的X_bar
        part_X_bar_list = X_bar_list[i:i + 15]
        count_7_num = 0
        for X_bar in part_X_bar_list:
            if np.absolute(X_bar - CL) < (xigama):
                count_7_num = count_7_num + 1
        if count_7_num > 14:
            flag = 1
            break
    return flag


def judge_8_type(data):
    # 判断数据是不是属于连续8点落在中心线两侧但无1点在C区域内
    R_list = [np.max(i) - np.min(i) for i in data]
    # 计算样本计算这25个样本点的CL、R_bar、UCL和LCL,σ
    X_bar_list = [np.mean(i) for i in data]
    CL = np.mean(X_bar_list)
    R_bar = np.mean(R_list)
    UCL = CL + 0.577 * R_bar
    LCL = CL - 0.577 * R_bar
    xigama = 1.154 * R_bar / 6
    # 对这段数据求导数，然后去连续8点落在中心线两侧但无1点在C区域内
    flag = 0
    for i in range(len(X_bar_list) - 8):
        # 截取长度为8的X_bar
        part_X_bar_list = X_bar_list[i:i + 8]
        count_8_num = 0
        for X_bar in part_X_bar_list:
            if np.absolute(X_bar - CL) > (xigama):
                count_8_num = count_8_num + 1
            else:
                break
        if count_8_num > 7:
            flag = 1
            break
    return flag


def get_multi_label(data):
    label = np.zeros((9,))
    if judge_2_type(data):
        label[2] = 1
    if judge_3_type(data):
        label[3] = 1
    if judge_4_type(data):
        label[4] = 1
    if judge_5_type(data):
        label[5] = 1
    if judge_6_type(data):
        label[6] = 1
    if judge_7_type(data):
        label[7] = 1
    if judge_8_type(data):
        label[8] = 1
    return label
