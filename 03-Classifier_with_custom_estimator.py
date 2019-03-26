import tensorflow as tf
import os


MODEL_NAME = '03-Classifier_with_custom_estimator'
TRAIN_FILE_PATH = 'data\\iris_training.csv'
EVAL_FILE_PATH = 'data\\iris_test.csv'
PREDICT_FILE_PATH = 'data\\iris_predict.csv'
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setpsa', 'Versicolor', 'Virginica']
BATCH_SIZE = 500
NUM_EPOCHS = 1000
tf.logging.set_verbosity(tf.logging.INFO)

# 1.input_fn生成函数
def generate_input_fn(file_name, mode=tf.estimator.ModeKeys.EVAL,
                      batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS):
    def input_fn():
        dataset = tf.data.TextLineDataset(file_name).skip(1)
        def _parse_line(line):
            if mode == tf.estimator.ModeKeys.PREDICT:
                line = tf.decode_csv(line, [[0.0]]*4)
                features = dict(zip(CSV_COLUMN_NAMES[:-1], line))
                return features
            else:
                line = tf.decode_csv(line, [[0.0]]*4+[[0]])
                features = dict(zip(CSV_COLUMN_NAMES, line))
                label = features.pop('Species')
                return features, label
        dataset = dataset.map(_parse_line)
        if mode != tf.estimator.ModeKeys.PREDICT:
            dataset = dataset.shuffle(batch_size)
            dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset
    print('Input_fn built completed!')
    return input_fn


# 2. 创建特征列
feature_columns = []
for _ in CSV_COLUMN_NAMES[:-1]:
    feature_columns.append(tf.feature_column.numeric_column(key=_))

# 3. 自定义Model并创建Estimator
def custom_model(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units, activation=tf.nn.relu)
    logits = tf.layers.dense(net, params['n_classes'], activation=None)
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids':predicted_classes,
            'probabilities':tf.nn.softmax(logits),
            'logits':logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
    metric = {'accuracy':accuracy}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metric)

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

classifier = tf.estimator.Estimator(model_fn=custom_model,
                                    model_dir='trained_model\\03-Classifier_with_custom_estimator',
                                    params={'feature_columns':feature_columns,
                                            'hidden_units':[10, 10],
                                            'n_classes': 3})

# 4. estimator训练
classifier.train(input_fn=generate_input_fn(TRAIN_FILE_PATH,
                                           mode=tf.estimator.ModeKeys.TRAIN))

# 5. estimator验证
eval_result = classifier.evaluate(input_fn=generate_input_fn(EVAL_FILE_PATH,
                                               mode=tf.estimator.ModeKeys.EVAL))
print(eval_result)

# 6. estimator预测
predict_result = classifier.predict(input_fn=generate_input_fn(PREDICT_FILE_PATH,
                                              mode=tf.estimator.ModeKeys.PREDICT))
for p in predict_result:
    print('{}\n'.format(p))