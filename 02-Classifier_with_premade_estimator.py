import tensorflow as tf
import pandas as pd

MODEL_NAME = '02-Classifier_with_premade_estimator'
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setpsa', 'Versicolor', 'Virginica']
BATCH_SIZE = 100
TRAIN_STEPS = 1000

# 0.下载数据，定义数据输出函数
train_csv_path = tf.keras.utils.get_file(fname='F:\\Github\\Estimator_exercise\\data\\'+TRAIN_URL.split('/')[-1],
                                         origin=TRAIN_URL)
test_csv_path = tf.keras.utils.get_file(fname='F:\\Github\\Estimator_exercise\\data\\'+TEST_URL.split('/')[-1],
                                        origin=TEST_URL)
train_df = pd.read_csv(train_csv_path, names=CSV_COLUMN_NAMES, header=0)
test_df = pd.read_csv(test_csv_path, names=CSV_COLUMN_NAMES, header=0)

train_x, train_y = train_df, train_df.pop('Species')
test_x, test_y = test_df, test_df.pop('Species')
tf.logging.set_verbosity(tf.logging.INFO)


# 1.定义input_fn
def train_input_fn(train_x, train_y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(train_x), train_y))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset


def eval_input_fn(test_x, test_y, batch_size):
    if test_y is None:
        input_tensor = dict(test_x)
    else:
        input_tensor = (dict(test_x), test_y)
    dataset = tf.data.Dataset.from_tensor_slices(input_tensor)
    dataset = dataset.batch(batch_size)

    return dataset


# 2. 定义Feature_columns
feature_columns = []
for key in train_x.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))

# 3. 定义estimator
estimator = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 10],
    n_classes=3,
    model_dir='trained_model\\{}\\'.format(MODEL_NAME)
)

# 4. 模型训练
estimator.train(
    input_fn=lambda:train_input_fn(train_x, train_y, BATCH_SIZE),
    steps=TRAIN_STEPS
)

# 5. 模型验证
eval_result = estimator.evaluate(
    input_fn=lambda:eval_input_fn(test_x, test_y, BATCH_SIZE))
print(eval_result)

# 6. 模型训练
