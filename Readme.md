## 代码说明
本Repo是个人学习tf.Estimator组件及其相关API时创建的包含数据的代码集；

使用的tensorflow版本是1.12.

- 01-Regression_from_csv.py 

从csv文件中读取数据，使用预置的Estimator做回归训练

- 02-Classifier_with_premade_estimator.py

下载IRIS数据集，然后使用预置Estimator做分类训练

- 03-Classifier_with_custom_estimator.py

使用IRIS数据集和Custom Estimator做分类训练

- 04-MNIST_classifier_with_premade_estimator.py

使用DNNClassifier完成MNIST分类

- 05-MNIST_classifier_with_custom_estimator.py

使用自制卷积神经网络完成MNIST分类


## 更新日志
#### 20190321
1. 在Github远程仓库创建Repo；
2. 导入数据集，完成01-Regression_from_csv.py

#### 20190322
1. 在01-Regression_from_csv.py中完成了Predict输出结果功能，并修改了原代码中的input_fn的EVAL模式问题，即在进行预测时无需再输入label；
2. 完成02-Classifier_with_premade_estimator.py；

#### 20190325
1. 补全02-Classifier_with_premade_estimator.py中预测部分代码；
2. 完成03-Classifier_with_custom_estimator.py模型构建和训练部分代码；

#### 20190326
1. 完成03-Classifier_with_custom_estimator.py的预测部分代码；
2. 完成04-MNIST_classifier_with_premade_estimator.py
3. 完成05-MNIST_classifier_with_custom_estimator.py


## 参考资料
- https://github.com/GoogleCloudPlatform/tf-estimator-tutorials
- https://github.com/tensorflow/models/blob/master/samples/core/get_started/iris_data.py
- https://www.tensorflow.org/guide/custom_estimators
- https://github.com/tensorflow/models/blob/master/samples/core/get_started/custom_estimator.py
- https://github.com/tensorflow/models/blob/master/samples/core/get_started/premade_estimator.py
- https://codeburst.io/use-tensorflow-dnnclassifier-estimator-to-classify-mnist-dataset-a7222bf9f940
- https://www.tensorflow.org/tutorials/estimators/cnn

