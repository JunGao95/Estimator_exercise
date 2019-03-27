
import tensorflow as tf
import numpy as np
BATCH_SIZE = 100
tf.logging.set_verbosity(tf.logging.INFO)
train, test = tf.keras.datasets.mnist.load_data()
train_x, train_y = train
# 1. 定义输入生成函数
def generate_input_fn(mode, batch_size=BATCH_SIZE):
    def input_fn():
        if mode == tf.estimator.ModeKeys.TRAIN:
            mnist_ds = tf.data.Dataset.from_tensor_slices(({'x': train[0].astype(np.float32).reshape((-1, 784))},
                                                train[1].astype(np.int32).reshape((-1, 1))))
            mnist_ds = mnist_ds.shuffle(60000).repeat().batch(batch_size)
            return mnist_ds
        elif mode == tf.estimator.ModeKeys.EVAL:
            mnist_ds = tf.data.Dataset.from_tensor_slices(({'x':test[0].astype(np.float32).reshape(-1, 784)},
                                                           test[1].astype(np.int32).reshape((-1, 1))))
            mnist_ds = mnist_ds.batch(batch_size)
            return mnist_ds
        else:
            assert mode == tf.estimator.ModeKeys.PREDICT
            mnist_ds = tf.data.Dataset.from_tensor_slices({'x':test[0].astype(np.float32).reshape(-1, 784)})
            mnist_ds = mnist_ds.batch(batch_size)
            return mnist_ds
    return input_fn

# 2. 特征列构建
feature_column = [tf.feature_column.numeric_column(key='x', shape=[784, ])]

# 3. 实例化Estimator
estimator = tf.estimator.DNNClassifier(hidden_units=[100, 100],
                                       feature_columns=feature_column,
                                       model_dir='trained_model\\04-MNIST_classifier_with_premade_estimator',
                                       n_classes=10)

# 4. 进行训练
estimator.train(input_fn=generate_input_fn(mode=tf.estimator.ModeKeys.TRAIN),
                steps=100)

# 5. 进行验证
eval_result = estimator.evaluate(input_fn=generate_input_fn(mode=tf.estimator.ModeKeys.EVAL))
print(eval_result)

# 6. 预测
predict_results = estimator.predict(input_fn=generate_input_fn(mode=tf.estimator.ModeKeys.PREDICT))
for p in predict_results:
    print(p)
