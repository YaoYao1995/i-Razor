#encoding=utf-8
from torch.utils.tensorboard import SummaryWriter
import wandb
import sys
import time
import os
import __init__
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sys.path.append(__init__.config['data_path']) # add your data path here
from datasets import as_dataset
from tf_trainer import Trainer, create_logger
from irazor_models import *
import traceback
import random
import numpy as np
import datetime
from glob import glob

data_name = 'avazu'
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
    tf.compat.v1.set_random_seed(seed)

def run_one_model(model=None,learning_rate=1e-3,decay_rate=1.0,epsilon=1e-8,ep=5, grda_c=0.005,
                  grda_mu=0.51, learning_rate2=1e-3, decay_rate2=1.0, retrain_stage=False, writer=None, logger=None):
    n_ep = ep * 2
    train_param = {
        'opt1': 'adagrad',
        'opt2': 'adagrad',
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
        'test_every_epoch': max(int(ep / 5),1),
        'retrain_stage': retrain_stage,
        'writer': writer,
        'logger': logger,
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
    seed_tensorflow(seed=1217)
    pretrain = True
    #config = "xDeepFM_config"
    #config = "DNN_config"
    config = "FDE_config"
    if pretrain:
        model_string_name = "DeepFM_Pretrain"
        model = DeepFMPretrainAndRetrain(init='xavier', num_inputs=dataset.max_length, input_emb_size_config=[30]*dataset.max_length, input_feature_min=dataset.feat_min, input_feat_num=dataset.feat_sizes, l2_weight=0.001, l2_bias=0.001, target_vec_sizes=[0,1,2,4,8,16,30], fid_loss_wt=1e-4, temperature=0.05,mlp=mlp, bn=False, mode="pretrain")
    else:
        if config == "DNN_config":
            model_string_name = "DeepFM_Retrain_DNN"
            avazu_emb_configs = [[1, 1], [2, 12], [3, 18], [5, 18], [6, 1], [7, 1], [8, 5], [9, 22], [10, 18], [12, 1], [13, 5], [16, 6], [19, 6], [20, 3], [21, 2], [22, 3]]
        elif config == "DeepFM_config":
            model_string_name = "DeepFM_Retrain"
            avazu_emb_configs = []
        else: 
            model_string_name = "DeepFM_Retrain_FDE"
            avazu_emb_configs = [[i, 30] for i in range(dataset.max_length)]
        emb_configs = avazu_emb_configs
        input_size_config =[0] * dataset.max_length
        total_dim = 0
        total_params = 0
        total_fieds = 0
        drop_list = list(range(dataset.max_length))
        for field,dim in emb_configs:
            input_size_config[field] = dim
            total_dim += dim
            total_params += dim * dataset.feat_sizes[field]
            total_fieds += 1
            drop_list.remove(field)
        print("**"*50)
        print(f"config-{config}, fileds-{total_fieds}, parms-{total_params}, dim-{total_dim}. Dropfields-{drop_list}.")
        print("**"*50)
        model = DeepFMPretrainAndRetrain(init='xavier', num_inputs=dataset.max_length, input_emb_size_config=input_size_config, input_feature_min=dataset.feat_min, input_feat_num=dataset.feat_sizes, l2_weight=0.001, l2_bias=0.001, temperature=0.05,mlp=mlp, bn=False, mode="retrain")
    # Setup an experiment folder:
    base_dir = "/home/ubuntu/results/"
    os.makedirs(base_dir + model_string_name, exist_ok=True)
    results_dir = os.path.join(base_dir, model_string_name, data_name)
    os.makedirs(results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{results_dir}/*"))    
    experiment_dir = f"{results_dir}/{experiment_index:03d}-{mlp}-bs-{batch_size}"  # Create an experiment folder
    #checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    #os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(experiment_dir+"/tf_log", exist_ok=True)
    writer = SummaryWriter(experiment_dir+"/tf_log")
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    logger.info(f"Batchsize: {batch_size}")
    if not pretrain:
        logger.info(f"config-{config}, fileds-{total_fieds}, parms-{total_params}, dim-{total_dim}. Dropfields-{drop_list}.")
    now=datetime.datetime.now()
    time_label = now.strftime("%Y-%m-%d %H:%M:%S")
    wandb.init(project="irazor", group=data_name+"-"+model_string_name, tags=str(batch_size), entity="yao-yao", dir="/workspace/wandb/", name=f"{data_name}-BS-{batch_size}-{experiment_index:03d}-{model_string_name}-"+time_label)
    # define a metric we are interested in the minimum of
    wandb.define_metric("test_log_loss", summary="min")
    wandb.define_metric("train_loss", summary="min")
    wandb.define_metric("train_l2_loss", summary="min")
    
    # define a metric we are interested in the maximum of
    wandb.define_metric("test_auc", summary="max")
    wandb.define_metric("train_moving_auc", summary="max")
    wandb.log({'batch_size': batch_size,})
    run_one_model(model=model, learning_rate=learning_rate, epsilon=1e-8,
                  decay_rate=None, ep=split_epoch, grda_c=None, grda_mu=None, 
                  learning_rate2=None,decay_rate2=None, retrain_stage=True,
                  writer=writer, logger=logger
                 )
    writer.close()
    logger.info("Done!")
    wandb.finish()

















