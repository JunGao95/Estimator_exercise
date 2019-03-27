import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)
train, test = tf.keras.datasets.mnist.load_data()

BATCH_SIZE = 100

# 1. 定义输入生成函数
def generate_input_fn(mode, batch_size=BATCH_SIZE):

    def input_fn():
        train_x, train_y = train
        test_x, test_y = test
        predict_x = test[0]

        if mode == tf.estimator.ModeKeys.TRAIN:
            mnist_ds = tf.data.Dataset.from_tensor_slices(({'x':train_x.astype(np.float32)},
                                                           train_y.astype(np.int32).reshape((-1, 1))))
            mnist_ds = mnist_ds.shuffle(60000).repeat().batch(batch_size)
            return mnist_ds
        elif mode == tf.estimator.ModeKeys.EVAL:
            mnist_ds = tf.data.Dataset.from_tensor_slices(({'x':test_x.astype(np.float32)},
                                                           test_y.astype(np.int32).reshape((-1, 1))))
            mnist_ds = mnist_ds.batch(batch_size)
            return mnist_ds
        elif mode == tf.estimator.ModeKeys.PREDICT:
            mnist_ds = tf.data.Dataset.from_tensor_slices({'x':predict_x.astype(np.float32)})
            mnist_ds = mnist_ds.batch(batch_size)
            return mnist_ds

    return input_fn

# 2.定义feature_column
feature_column = [tf.feature_column.numeric_column('x', (28, 28))]

# 3.自定义Model并实例化
def cnn_model_fn(features, labels, mode):
    input_layer = tf.feature_column.input_layer(features, feature_column)
    input_layer = tf.reshape(input_layer, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )
    # 输出形为[batch_size, 28, 28, 32]的张量
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # 输出形为[batch_size, 14, 14, 32]的张量
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )
    # 输出形为[batch_size, 14, 14, 64]的张量
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # 输出形为[batch_size, 7, 7, 64]的张量
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    # 输出形为[batch_size, 3136]的张量
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        'classes':tf.argmax(input=logits, axis=1),
        'probabilities':tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if mode==tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    assert mode == tf.estimator.ModeKeys.EVAL
    eval_metric_op = {
        'accuracy':tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes']
        )
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_op)

estimator = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                   model_dir='trained_model\\05-MNIST_classifier_with_custom_estimator')

# 4. 模型训练
estimator.train(input_fn=generate_input_fn(mode=tf.estimator.ModeKeys.TRAIN),
                steps=200)

# 5. 模型验证
eval_result = estimator.evaluate(input_fn=generate_input_fn(mode=tf.estimator.ModeKeys.EVAL))
print(eval_result)

# 6. 预测
predict_result = estimator.predict(input_fn=generate_input_fn(mode=tf.estimator.ModeKeys.PREDICT))
for p in predict_result:
    print(p)