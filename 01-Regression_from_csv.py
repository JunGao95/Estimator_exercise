import tensorflow as tf
import numpy as np
import pandas as pd

HEADERS = ['key', 'x', 'y', 'alpha', 'beta', 'target']
NUMERIC_FEATURE_NAMES = ['x', 'y']
CATEGORICAL_FEATURE_DICT = {'alpha':['ax01', 'ax02'],
                            'beta' :['bx01', 'bx02']}
CATEGORICAL_FEATURE_NAME = list(CATEGORICAL_FEATURE_DICT.keys())
FEATURE_NAME = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAME


# 1.定义导入函数，从csv文件中返回一个input_fn
def generate_pandas_input_fn(file_name,
                             mode=tf.estimator.ModeKeys.EVAL,
                             num_epochs=1,
                             batch_size=100):
    train_df = pd.read_csv(file_name,
                           names=HEADERS)
    x = train_df[FEATURE_NAME].copy()
    y = train_df['target']

    num_epochs = num_epochs if mode==tf.estimator.ModeKeys.TRAIN else 1
    shuffle = True if mode==tf.estimator.ModeKeys.EVAL else False

    pandas_input_fn = tf.estimator.inputs.pandas_input_fn(x=x,
                                                          y=y,
                                                          batch_size=batch_size,
                                                          num_epochs=num_epochs,
                                                          shuffle=shuffle,
                                                          target_column='target'
                                                          )
    print("data input fn:")
    print("Input_file:{}".format(file_name))
    print("Dataset_size:{}".format(len(train_df)))
    print("Num_epochs:{}".format(num_epochs))
    print("batch_size:{}".format(batch_size))

    return pandas_input_fn


features, target = generate_pandas_input_fn(file_name=r'data\train-data.csv')()
print("Features read from csv:{}".format(list(features.keys())))
print("Target read from csv:{}").format(target)
print(target)


# 2.定义特征列函数
def get_feature_columns():
    feature_columns = {}
    for feature in NUMERIC_FEATURE_NAMES:
        feature_columns[feature] = feature_columns.append(tf.feature_column.numeric_column(key=feature))
    for item in CATEGORICAL_FEATURE_DICT.items():
        feature_columns[item[0]] = (tf.feature_column.categorical_column_with_vocabulary_list(key=item[0],
                                                                                              vocabulary_list=item[1]))
    feature_columns['alphaXbeta'] = tf.feature_column.crossed_column([feature_columns['alpha'], feature_columns['beta']], 4)
    print("Feature_columns: {}".format(feature_columns.keys()))
    return feature_columns


# 3.创建estimator
def create_estimator(run_config, hparams):


