#encoding=utf-8
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
import sys
import time
import os
import __init__
sys.path.append(__init__.config['data_path']) # add your data path here
from datasets import as_dataset
from tf_trainer import Trainer
from irazor_models import *
import traceback
import random
import numpy as np

data_name = 'avazu_demo'
dataset = as_dataset(data_name)
backend = 'tf'
batch_size = 128

train_data_param = {
    'gen_type': 'train',
    'random_sample': True,
    'batch_size': batch_size,
    'split_fields': False,
    'on_disk': True,
    'squeeze_output': True,
}
test_data_param = {
    'gen_type': 'test',
    'random_sample': False,
    'batch_size': batch_size,
    'split_fields': False,
    'on_disk': True,
    'squeeze_output': True,
}

def seed_tensorflow(seed=1217):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def run_one_model(model=None,learning_rate=1e-3,decay_rate=1.0,epsilon=1e-8,ep=5, grda_c=0.005,
                  grda_mu=0.51, learning_rate2=1e-3, decay_rate2=1.0, retrain_stage=0):
    n_ep = ep * 1
    train_param = {
        'opt1': 'adam',
        'opt2': 'adam',
        'loss': 'weight',
        'pos_weight': 1.0,
        'n_epoch': n_ep,
        'train_per_epoch': dataset.train_size / ep,  # split training data
        'test_per_epoch': dataset.test_size,
        'early_stop_epoch': int(0.5*ep),
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'decay_rate': decay_rate,
        'learning_rate2': learning_rate2,
        'decay_rate2': decay_rate2,
        'epsilon':epsilon,
        'load_ckpt': False,
        'ckpt_time': 10000,
        'grda_c': grda_c,
        'grda_mu': grda_mu,
        'test_every_epoch': int(ep / 5),
        'retrain_stage': retrain_stage,
    }
    train_gen = dataset.batch_generator(train_data_param)
    test_gen = dataset.batch_generator(test_data_param)
    trainer = Trainer(model=model, train_gen=train_gen, test_gen=test_gen, **train_param)
    trainer.fit()
    trainer.session.close()


import math
if __name__=="__main__":
    # general parameter
    learning_rate = 0.01
    split_epoch = 5
    mlp = [700]*5+[1]
    mlp = [10]*5+[1]
    #feature_result_configs for irazor/Darts/Autodim, here we give the results of iRazor as an example.
    avazu_emb_configs = [[2, 1], [3, 12], [4, 18], [6, 18], [7, 1], [8, 1], [9, 5], [10, 22], [11, 18], [13, 1], [14, 5], [17, 6], [20, 6], [21, 3], [22, 2], [23, 3]]
    criteo_emb_configs = [[1, 15], [2, 12], [3, 4], [4, 17], [5, 4], [6, 4], [7, 4], [8, 17], [9, 2], [10, 4], [11, 12], [12, 24], [13, 13], [14, 8], [15, 24], [16, 25], [19, 4], [20, 24], [21, 16], [22, 16], [23, 17], [24, 25], [25, 21], [26, 17], [27, 8], [28, 23], [29, 4], [30, 8], [32, 16], [33, 4], [34, 21], [35, 8], [36, 23], [37, 23], [38, 17], [39, 23]]

    emb_configs = avazu_emb_configs
    input_size_config =[0]* dataset.max_length
    for field,dim in emb_configs:
        input_size_config[field] = dim
    seed_tensorflow(seed=1217)
    model = DNNRetrain(init='xavier', num_inputs=dataset.max_length, input_emb_size_config=input_size_config, input_feature_min=dataset.feat_min, input_feat_num=dataset.feat_sizes, l2_weight=0.001, l2_bias=0.001,
                 mlp=mlp, bn=False, ln=True)
    run_one_model(model=model, learning_rate=learning_rate, epsilon=1e-8,
                  decay_rate=None, ep=split_epoch, grda_c=None, grda_mu=None, 
                  learning_rate2=None,decay_rate2=None, retrain_stage=None)

















