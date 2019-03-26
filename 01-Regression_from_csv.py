import tensorflow as tf


MODEL_NAME = '01_Regression_from_csv'
HEADERS = ['key', 'x', 'y', 'alpha', 'beta', 'target']
EVAL_HEADERS = ['key', 'x', 'y', 'alpha', 'beta']
NUMERIC_FEATURE_NAMES = ['x', 'y']
INDICATOR_FEATURE_NAMES = []
CATEGORICAL_FEATURE_DICT = {'alpha':['ax01', 'ax02'],
                            'beta' :['bx01', 'bx02']}
CATEGORICAL_FEATURE_NAME = list(CATEGORICAL_FEATURE_DICT.keys())
FEATURE_NAME = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAME
TRAIN_PATH = 'data\\train-data.csv'
TEST_PATH = 'data\\test-data.csv'
VALID_PATH = 'data\\valid-data.csv'


# 1.定义导入函数，从csv文件中返回
def generate_csv_input_fn(file_name,
                             mode=tf.estimator.ModeKeys.EVAL,
                             num_epochs=2,
                             batch_size=500):
    def input_fn(file_name=file_name, mode=mode, num_epochs=num_epochs, batch_size=batch_size):
        ds = tf.data.TextLineDataset(file_name)
        def _parse_line(line):
            if mode == tf.estimator.ModeKeys.PREDICT:
                fields = tf.decode_csv(line, [[0], [0.0], [0.0], ['ax01'], ['bx01']])
                features = dict(zip(EVAL_HEADERS, fields))
                return features
            else:
                fields = tf.decode_csv(line, [[0], [0.0], [0.0], ['ax01'], ['bx01'], [0.0]])
                features = dict(zip(HEADERS, fields))
                label = features.pop('target')
                return features, label
        ds = ds.map(_parse_line)

        num_epochs = num_epochs if mode==tf.estimator.ModeKeys.TRAIN else 1
        shuffle = False if mode==tf.estimator.ModeKeys.PREDICT else True
        ds = ds.repeat(num_epochs)
        if shuffle:
            ds = ds.shuffle(10000)
        ds = ds.batch(batch_size)

        return ds

    print("Input_fn building completed")

    return input_fn


# 2.定义特征列函数
def get_feature_columns():
    feature_columns = {}
    for feature in NUMERIC_FEATURE_NAMES:
        feature_columns[feature] = tf.feature_column.numeric_column(key=feature)
    for item in CATEGORICAL_FEATURE_DICT.items():
        feature_columns[item[0]] = tf.feature_column.categorical_column_with_vocabulary_list(key=item[0],
                                                                                             vocabulary_list=item[1]
                                                                                             )
    feature_columns['alphaXbeta'] = tf.feature_column.crossed_column([feature_columns['alpha'], feature_columns['beta']], 4)
    CATEGORICAL_FEATURE_NAME.append('alphaXbeta')
    for _ in CATEGORICAL_FEATURE_NAME:
        feature_columns['indicator_{}'.format(_)] = tf.feature_column.indicator_column(feature_columns[_])
        INDICATOR_FEATURE_NAMES.append('indicator_{}'.format(_))
    feature_columns['indicator_alphaXbeta'] = tf.feature_column.indicator_column(feature_columns['alphaXbeta'])
    estimator_feature_columns = list(feature_columns[_] for _ in INDICATOR_FEATURE_NAMES)
    print("Estimator_feature_columns: {}".format(estimator_feature_columns))
    print("Feature_columns: {}".format(feature_columns.keys()))

    return estimator_feature_columns


# 3.创建estimator
def create_estimator(hparams):

    feature_columns = get_feature_columns()
    estimator = tf.estimator.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=hparams.hidden_units,
        optimizer=tf.train.AdamOptimizer(),
        activation_fn=tf.nn.elu,
        dropout=hparams.dropout_prob,
        model_dir='trained_model\\{}'.format(MODEL_NAME)
    )

    print('Estimator built completed.')
    return estimator


hparams = tf.contrib.training.HParams(
    num_epochs = 100,
    batch_size = 500,
    hidden_units = [8,4],
    dropout_prob = 0.0
)


# 4.实例化Estimator并训练
estimator = create_estimator(hparams)

train_input_fn = generate_csv_input_fn(file_name='data\\train-data.csv',
                                          mode=tf.estimator.ModeKeys.TRAIN,
                                          num_epochs=hparams.num_epochs,
                                          batch_size=hparams.batch_size)
tf.logging.set_verbosity(tf.logging.INFO)
print("Estimator training started.")
estimator.train(input_fn = train_input_fn)
print("Estimator training finished")

# 5.在测试集上训练
test_input_fn = generate_csv_input_fn(TEST_PATH,
                                         tf.estimator.ModeKeys.EVAL,
                                         batch_size = 5000)
result = estimator.evaluate(input_fn=test_input_fn)
print(result)

# 6.获得预测结果
predict_input_fn = generate_csv_input_fn(file_name=VALID_PATH,
                                         mode=tf.estimator.ModeKeys.PREDICT,
                                         batch_size=3000)
predictions = estimator.predict(input_fn=predict_input_fn)
print("Prediction finished!")
pre_list = list(predictions)
print(pre_list)
