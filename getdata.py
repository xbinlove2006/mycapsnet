


import numpy
SEGMENTS_BASE_PATH = "E:/Django_study/pro_test/App/ecgprocess/ecgsets/"


# 从文件中读取数据
def get_train_data():
    train=numpy.load(SEGMENTS_BASE_PATH+'train.npy',allow_pickle=True)
    traintag=numpy.load(SEGMENTS_BASE_PATH+'traintag.npy',allow_pickle=True)
    return train,traintag
    pass
def get_test_data():
    test=numpy.load(SEGMENTS_BASE_PATH+'test.npy',allow_pickle=True)
    testtag=numpy.load(SEGMENTS_BASE_PATH+'testtag.npy',allow_pickle=True)
    return test,testtag
    pass