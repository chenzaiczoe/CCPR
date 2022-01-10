from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import hamming_loss
from sklearn.ensemble import RandomForestClassifier
from keras.layers import Dense, Input, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D, concatenate
from keras import activations, models, optimizers, losses
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_feature_data(all_data):
    DATA = []
    for i in range(all_data.shape[0]):
        data = all_data[i]
        X_bar_list = [np.mean(j) for j in data]
        DATA.append(X_bar_list)
    return np.array(DATA)


def DNN_model():
    nclass = 7
    inp = Input(shape=(25,), name="input_1")
    dense_1 = Dense(64, activation=activations.relu)(inp)
    dense_1 = Dense(64, activation=activations.sigmoid)(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)
    model = models.Model(inputs=inp, outputs=dense_1)
    return model


if __name__ == '__main__':
    """
    读取数据
    """
    sc = StandardScaler()
    data_train = get_feature_data(np.load(r'../data_train.npy'))
    # data_train=sc.fit_transform(data_train)
    label_train = to_categorical(np.load(r'../label_train.npy'))

    data_test = get_feature_data(np.load(r'../data_test.npy'))
    # data_test =sc.fit_transform(data_test)
    label_test = np.load(r'../label_test.npy')
    """
    训练模型
    """
    print('开始预测')
    # 使用支持随机森林
    clf = RandomForestClassifier().fit(data_train, label_train)
    y_pred = clf.predict(data_test)
    y_pred = np.argmax(y_pred, axis=1)
    RD_confusion_matrix = pd.crosstab(label_test, y_pred, rownames=['label'], colnames=['predict'])
    print(RD_confusion_matrix)
    RD_accuracy_score = accuracy_score(label_test, y_pred)
    print(f'随机森林 acc:{RD_accuracy_score}')
    #
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

    # 使用神经网络
    file_path = r'DNN.h5'  # 设定存贮路径
    model = DNN_model()
    model.summary()
    opt = optimizers.Adam(0.00001)
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=20, verbose=2)
    callbacks_list = [checkpoint, early, redonplat]  # early
    # model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    model.compile(optimizer=optimizers.SGD(0.0001), loss=losses.mse, metrics=['acc'])
    model.fit(x=data_train,
              y=label_train,
              batch_size=16,
              epochs=100,
              verbose=2,
              callbacks=callbacks_list,
              validation_split=0.1)
    y_pred = model.predict(data_test)
    y_pred = np.argmax(y_pred, axis=1)
    DNN_confusion_matrix = pd.crosstab(label_test, y_pred, rownames=['label'], colnames=['predict'])
    print(DNN_confusion_matrix)
    DNN_accuracy_score = accuracy_score(label_test, y_pred)
    print(f'DNN acc:{DNN_accuracy_score}')
