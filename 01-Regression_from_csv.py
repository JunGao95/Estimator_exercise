import tensorflow as tf
import numpy as np
import pandas as pd



def generate_pandas_input_fn(file_name,
                             mode=tf.estimator.ModeKeys.EVAL,
                             num_epoches=1,
                             batch_size=100):
    train_df = pd.read_csv('data\\train-data.csv')


