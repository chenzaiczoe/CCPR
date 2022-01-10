from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import hamming_loss
from sklearn.ensemble import RandomForestClassifier
from keras.layers import Dense, Input, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D, concatenate
from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd


def get_feature_data(all_data):
    DATA = []
    for i in range(all_data.shape[0]):
        data = all_data[i]

        R_list = [np.max(i) - np.min(i) for i in data]
        # 计算样本计算这25个样本点的CL、R_bar、UCL和LCL
        X_bar_list = [np.mean(i) for i in data]
        CL = np.mean(X_bar_list)
        R_bar = np.mean(R_list)
        UCL = CL + 0.577 * R_bar
        LCL = CL - 0.577 * R_bar
        # feature_list,设定需要提取的特征
        feature_list = [i for i in X_bar_list]
        feature_list.append(CL)
        feature_list.append(R_bar)
        feature_list.append(UCL)
        feature_list.append(LCL)
        DATA.append(feature_list)
    return np.array(DATA)


if __name__ == '__main__':
    # data_train = get_feature_data(np.load(r'data_train.npy'))
    data_train = get_feature_data(np.load(r'../data_train.npy'))
    label_train = to_categorical(np.load(r'../label_train.npy'))

    data_val = get_feature_data(np.load(r'../data_val.npy'))
    label_val = to_categorical(np.load(r'../label_val.npy'))

    data_test = get_feature_data(np.load(r'../data_test.npy'))
    label_test = np.load(r'../label_test.npy')

    print('开始预测')
    # 使用支持随机森林
    clf = RandomForestClassifier().fit(data_train, label_train)
    y_pred = clf.predict(data_test)
    y_pred = np.argmax(y_pred, axis=1)
    RD_confusion_matrix = pd.crosstab(label_test, y_pred, rownames=['label'], colnames=['predict'])
    print(RD_confusion_matrix)
    RD_accuracy_score = accuracy_score(label_test, y_pred)
    print(f'随机森林 acc:{RD_accuracy_score}')

    # 使用KNN分类器
    clf = KNeighborsClassifier().fit(data_train, label_train)
    y_pred = clf.predict(data_test)
    y_pred = np.argmax(y_pred, axis=1)
    KNN_confusion_matrix = pd.crosstab(label_test, y_pred, rownames=['label'], colnames=['predict'])
    print(KNN_confusion_matrix)
    KNN_accuracy_score = accuracy_score(label_test, y_pred)
    print(f'使用KNN分类器 acc:{KNN_accuracy_score}')

    # 使用支持向量机
    clf = SVC(decision_function_shape='ovo').fit(data_train, np.argmax(label_train, axis=1))
    y_pred = clf.predict(data_test)
    # y_pred = np.argmax(y_pred, axis=1)
    SVC_confusion_matrix = pd.crosstab(label_test, y_pred, rownames=['label'], colnames=['predict'])
    print(SVC_confusion_matrix)
    SVC_accuracy_score = accuracy_score(label_test, y_pred)
    print(f'支持向量机 acc:{SVC_accuracy_score}')
