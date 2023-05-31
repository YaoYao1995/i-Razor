from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from abc import abstractmethod
from itertools import combinations
import numpy as np
from tf_utils import normal_mlp

import __init__
from tf_utils import emb_lookup_multi_emb_size, row_col_fetch, row_col_expand, batch_kernel_product, \
    batch_mlp, create_placeholder, drop_out, embedding_lookup, linear, output, bin_mlp, get_variable, \
    layer_normalization, batch_normalization, get_l2_loss, split_data_mask

dtype = __init__.config['dtype']

if dtype.lower() == 'float32' or dtype.lower() == 'float':
    dtype = tf.float32
elif dtype.lower() == 'float64':
    dtype = tf.float64

class Model:
    inputs = None
    outputs = None
    logits = None
    labels = None
    learning_rate = None
    loss = None
    l2_loss = None
    optimizer = None
    grad = None

    @abstractmethod
    def compile(self, **kwargs):
        pass

    def __str__(self):
        return self.__class__.__name__

def generate_pairs(ranges=range(1, 100), mask=None, order=2):
    res = []
    for i in range(order):
        res.append([])
    for i, pair in enumerate(list(combinations(ranges, order))):
        if mask is None or mask[i]==1:
            for j in range(order):
                res[j].append(pair[j])
    print("generated pairs", len(res[0]))
    return res

def alloc_emb_for_irazor(embed_list, target_vec_sizes=[0, 1, 2, 6], temp=1.0, feature_num=[],training=None):
    #print(embed_list)
    print("search space", target_vec_sizes)
    sum_feature_num = sum(feature_num)
    #feature_proportion = [ _*1.0 /sum_feature_num for _ in feature_num ]
    #feature_proportion = [ 1.0 / len(feature_num)] * len(feature_num)
    feature_proportion = [ 0.01] * len(feature_num)
    max_size = max(target_vec_sizes)
    masks = []
    mask_valid_sizes = []
    for i, mask_size in enumerate(target_vec_sizes):
        if i > 0:
            cur_mask = [[0.] * target_vec_sizes[i - 1] + [1.0] * (
                    mask_size - target_vec_sizes[i - 1]) + [0.0] * (
                                max_size - mask_size)]
            mask_valid_sizes.append([mask_size - target_vec_sizes[i - 1]])
        else:
            cur_mask = [[1.0] * mask_size + [0.0] * (max_size - mask_size)]
            mask_valid_sizes.append([mask_size])
        masks.append(cur_mask)  # 1* emb
    mask_size_vec = tf.constant(mask_valid_sizes, dtype=tf.float32)  # ops*1
    import numpy as np
    masks = np.concatenate(masks, axis=0) # ops  * max_size
    total_mask = tf.constant(masks, name="masks", dtype=tf.float32) # ops*max_size
    print("masks", masks)
    fid_proportion = []
    used_embs = []
    for i in range(len(embed_list)):
        vec = embed_list[i]
        after_bn_vec = tf.layers.batch_normalization(vec, axis=-1, 
                                                reuse=tf.AUTO_REUSE, scale=False, center=False, name='prune_bn')
        used_embs.append(after_bn_vec)
        #used_embs.append(vec)
    fid_proportion_vec = tf.constant([feature_proportion], dtype=tf.float32)  # 1* input_features
    embedding = tf.stack(used_embs, axis=1,
                        name="bn_embeds")  # BN * input_features * emb
    logits = get_variable(init_type="zero",shape=[len(embed_list), len(target_vec_sizes)], name="nas_chioce_logits")  # input_features * ops

    choice_probs = tf.nn.softmax(logits / temp, axis=1,
                                    name="nas_choice_prob")  # inputs_features * ops

    fid_loss = tf.squeeze(tf.matmul(fid_proportion_vec, tf.matmul(choice_probs, mask_size_vec)), name="fid_loss")
    choice_matrix = tf.matmul(choice_probs, total_mask, # 
                                name= "choice_probs")  # input_features * emb
    output_embs = tf.expand_dims(choice_matrix, axis=0) * embedding # bs * input_features * emb 
    print("output_embs: ",output_embs)
    print("choice_probs: ",choice_probs)
    print("logits: ",logits)
    return tf.layers.flatten(output_embs), logits, fid_loss

def alloc_emb_for_autofield(embed_list, target_vec_sizes=[0, 30], temp=1.0, training=None):
    #print(embed_list)
    max_size = max(target_vec_sizes)
    masks = []
    mask_valid_sizes = []
    for i, mask_size in enumerate(target_vec_sizes):
        if i > 0:
            cur_mask = [[1.0] * max_size]
        else:
            cur_mask = [[0.0] * max_size]
        masks.append(cur_mask)  # 1* emb
    import numpy as np
    masks = np.concatenate(masks, axis=0) # ops  * max_size
    total_mask = tf.constant(masks, name="masks", dtype=tf.float32) # 2*max_size
    print("masks", masks)
    used_embs = []
    for i in range(len(embed_list)):
        vec = embed_list[i]
        after_bn_vec = tf.layers.batch_normalization(vec, axis=-1, training=training,
                                                reuse=tf.AUTO_REUSE, scale=False, center=False, name='prune_bn')
        used_embs.append(after_bn_vec)
       
    embedding = tf.stack(used_embs, axis=1, name="bn_embeds")  # BN * input_features * emb
    logits = get_variable(init_type="zero",shape=[len(embed_list), 2], name="nas_chioce_logits")  # input_features * ops

    choice_probs = tf.nn.softmax(logits / temp, axis=1, name="nas_choice_prob")  # inputs_features * 2

    choice_matrix = tf.matmul(choice_probs, total_mask, name= "choice_probs")  # input_features * emb
    output_embs = tf.expand_dims(choice_matrix, axis=0) * embedding # bs * input_features * emb 
    return tf.layers.flatten(output_embs)

def alloc_emb_for_autodim(embed_list, target_vec_sizes=[1, 2,4, 6], training=None, global_step=None):
    max_size = max(target_vec_sizes)
    masks = []
    for i, mask_size in enumerate(target_vec_sizes):
        cur_mask = [[1]*mask_size +  [0.0] * (max_size - mask_size)]
        masks.append(cur_mask)  # 1* emb
    import numpy as np
    masks = np.concatenate(masks, axis=0) # ops  * max_size
    total_mask = tf.constant(masks, name="masks", dtype=tf.float32) # ops*max_size
    used_embs = []
    for i in range(len(embed_list)):
        vec = embed_list[i]
        after_bn_vec = tf.layers.batch_normalization(vec, axis=-1, training=training,
                                                reuse=tf.AUTO_REUSE, scale=False, center=False, name='prune_bn')
        used_embs.append(after_bn_vec)

    embedding = tf.stack(used_embs, axis=1,
                        name="bn_embeds")  # BN * input_features * emb
    logits = get_variable(init_type="zero",shape=[len(embed_list), len(target_vec_sizes)], name="nas_chioce_logits")
    u = tf.random_uniform([len(embed_list), len(target_vec_sizes)], minval=0, maxval=1.0)
    gumbel_noise = -tf.log(-tf.log(u))
    cur_progress = tf.cast(global_step, tf.float32)
    temp  = tf.minimum(tf.maximum(0.01, 1-0.00005*cur_progress),0.2)
    
    choice_probs = tf.nn.softmax(logits, axis=1, name="nas_choice_prob")
    used_logits = (tf.log(choice_probs) + gumbel_noise) /temp
    used_choice_probs = tf.nn.softmax(used_logits, axis=1, name="used_choice_probs")
    
    choice_matrix = tf.matmul(used_choice_probs, total_mask,
                                name= "choice_probs")  # input_features * emb
    output_embs = tf.expand_dims(choice_matrix, axis=0) * embedding # bs * input_features * emb 

    return tf.layers.flatten(output_embs)

def alloc_emb_for_adafs(embed_list, target_vec_sizes=[30], training=None, temp=1.0):
    max_size = max(target_vec_sizes)
    used_embs = []
    for i in range(len(embed_list)):
        vec = embed_list[i]
        after_bn_vec = tf.layers.batch_normalization(vec, axis=-1, training=training,
                                                reuse=tf.AUTO_REUSE, scale=False, center=False, name='prune_bn')
        used_embs.append(after_bn_vec)

    embedding = tf.stack(used_embs, axis=1, name="bn_embeds")  # BN * input_features * emb
    logits = get_variable(init_type="zero",shape=[len(embed_list), 1], name="nas_chioce_logits")
    choice_probs = tf.nn.softmax(logits / temp, axis=0, name="nas_choice_prob")  # inputs_features * 1

    # choice_matrix = tf.tile(choice_probs, multiples=[1, max_size], name= "choice_probs")  # input_features * emb
    # output_embs = tf.expand_dims(choice_matrix, axis=0) * embedding # bs * input_features * emb 
    
    output_embs = tf.expand_dims(choice_probs, axis=0) * embedding # bs * input_features * emb 

    return tf.layers.flatten(output_embs)

def alloc_emb_for_darts(embed_list, target_vec_sizes=[1, 2,4, 6], training=None, global_step=None):
    max_size = sum(target_vec_sizes)
    masks = []
    mask_valid_sizes = []
    for i, mask_size in enumerate(target_vec_sizes):
        if i==0:
            before_num =0
        else:
            before_num = sum(target_vec_sizes[:i])
        cur_mask = [[0]*before_num + [1]*mask_size + [0.0] * (max_size - mask_size-before_num)]
        mask_valid_sizes.append([mask_size - target_vec_sizes[i - 1]])
        masks.append(cur_mask)  # 1* emb
    import numpy as np
    masks = np.concatenate(masks, axis=0) # ops  * max_size
    total_mask = tf.constant(masks, name="masks", dtype=tf.float32) # ops*max_size
    used_embs = []
    for i in range(len(embed_list)):
        vec = embed_list[i]
        after_bn_vec = tf.layers.batch_normalization(vec, axis=-1, training=training,
                                                reuse=tf.AUTO_REUSE, scale=False, center=False, name='training')
        used_embs.append(after_bn_vec)

    embedding = tf.stack(used_embs, axis=1,
                        name="bn_embeds")  # BN * input_features * emb
    logits = get_variable(init_type="zero",shape=[len(embed_list), len(target_vec_sizes)], name="nas_chioce_logits")
    u = tf.random_uniform([len(embed_list), len(target_vec_sizes)], minval=0, maxval=1.0)
    choice_probs = tf.nn.softmax(logits, axis=1, name="nas_choice_prob")
    
    choice_matrix = tf.matmul(choice_probs, total_mask,
                                name= "choice_probs")  # input_features * emb
    output_embs = tf.expand_dims(choice_matrix, axis=0) * embedding # bs * input_features * emb 

    return tf.layers.flatten(output_embs)

class DNNRetrain(Model):
    def __init__(self, init='xavier', num_inputs=None, input_emb_size_config=[], input_feature_min=[],input_feat_num=[], l2_weight=None, l2_bias=None,mlp=[], bn=False, ln=False):
        self.l2_weight = l2_weight
        self.l2_bias = l2_bias
        self.inputs, self.labels, self.training = create_placeholder(num_inputs, tf, True)
        assert num_inputs == len(input_emb_size_config) and num_inputs == len(input_feature_min) and num_inputs==len(input_feat_num)
        print("allocate all embs")
        all_embs = emb_lookup_multi_emb_size(self.inputs, input_feature_min,input_emb_size_config,input_feat_num)
        self.raw_embedding = tf.concat(all_embs, axis=1)
        print("allocate all bias")
        all_bias = emb_lookup_multi_emb_size(self.inputs, input_feature_min,[1 if ele>0 else 0 for ele in input_emb_size_config], input_feat_num)
        #all_bias = emb_lookup_multi_emb_size(self.inputs, input_feature_min,[1]*len(input_emb_size_config), input_feat_num)
        self.raw_bias = tf.concat(all_bias, axis=1)
        bias_sum = tf.add_n(all_bias) # bs *1
        output, self.layer_kernels, _ = normal_mlp(init="xavier", layer_sizes=mlp, layer_acts=["relu"]*(len(mlp)-1)+["none"], h=self.raw_embedding, node_in=int(self.raw_embedding.shape[1]), batch_norm=bn, layer_norm=ln, training=self.training)
        
        self.logits = tf.reduce_sum(output, axis=1) + tf.reduce_sum(bias_sum, axis=1) # bs
        self.outputs = tf.nn.sigmoid(self.logits)


    def compile(self, loss=None, optimizer1=None, optimizer2=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
                _loss_ = self.loss
                if not self.l2_weight:
                    self.l2_loss = get_l2_loss([1e-6, 1e-6],
                                               [self.raw_embedding, self.raw_bias])
                    _loss_ += self.l2_loss * 0
                else:
                    self.l2_loss = get_l2_loss([self.l2_weight, self.l2_bias],
                                               [self.raw_embedding, self.raw_bias])
                    _loss_ += self.l2_loss
                
                all_variable = [v for v in tf.trainable_variables()]
                self.optimizer1 = optimizer1.minimize(loss=_loss_, var_list=all_variable, global_step=global_step)

class AdafsPretrain(Model):
    def __init__(self, init='xavier', num_inputs=None, input_emb_size_config=[], input_feature_min=[],input_feat_num=[], l2_weight=None, l2_bias=None,
                  target_vec_sizes=[1,2,4,6], temperature=0.50, mlp=[], bn=False, ln=False):
        self.l2_weight = l2_weight
        self.l2_bias = l2_bias
        self.num_inputs = num_inputs
        self.search_space = target_vec_sizes
        self.inputs, self.labels, self.training = create_placeholder(num_inputs, tf, True)
        assert num_inputs == len(input_emb_size_config) and num_inputs == len(input_feature_min) and num_inputs==len(input_feat_num)

        all_embs = emb_lookup_multi_emb_size(self.inputs, input_feature_min,input_emb_size_config,input_feat_num)
        #print("all_embs:",all_embs)
        self.raw_embedding = tf.concat(all_embs, axis=1)
        
        emb_feature_for_concat = alloc_emb_for_adafs(embed_list=all_embs, target_vec_sizes=target_vec_sizes, temp=temperature,training=self.training)
        all_bias = emb_lookup_multi_emb_size(self.inputs, input_feature_min,[1]*len(input_emb_size_config), input_feat_num)

        self.raw_bias = tf.concat(all_bias, axis=1)
        bias_sum = tf.add_n(all_bias) # bs *1
        output, self.layer_kernels, _ = normal_mlp(init="xavier", layer_sizes=mlp, layer_acts=["relu"]*(len(mlp)-1)+["none"], h=emb_feature_for_concat, node_in=int(emb_feature_for_concat.shape[1]), batch_norm=bn, layer_norm=ln, training=self.training)
        print("mlp output", output,"bias_sum",bias_sum) # bs*1
        self.logits = tf.reduce_sum(output+bias_sum, axis=1,keep_dims=False) # bs
        print("lgits", self.logits)
        self.outputs = tf.nn.sigmoid(self.logits)


    def compile(self, loss=None, optimizer1=None, optimizer2=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
                _loss_ = self.loss
                if not self.l2_weight:
                    self.l2_loss = get_l2_loss([1e-6, 1e-6],
                                               [self.raw_embedding, self.raw_bias])
                    _loss_ += self.l2_loss * 0
                else:
                    self.l2_loss = get_l2_loss([self.l2_weight, self.l2_bias],
                                               [self.raw_embedding, self.raw_bias])
                    _loss_ += self.l2_loss
                all_variable = [v for v in tf.trainable_variables()]
                self.optimizer1 = optimizer1.minimize(loss=_loss_, var_list=all_variable, global_step=global_step)
                self.optimizer2 = None
                
                
    def analyse_structure(self, sess, print_full_weight=False, epoch=None, writer=None, logger=None):
        probs = sess.run(["nas_choice_prob:0"])[0]
        print("cur_nas_choice_prob:")
        probs = np.around(probs,4)
        for i in range(len(probs)):
            print(f"feature-{round(i,2)}: {probs[i][0]}")
        if logger is not None:
            logger.info(f"cur_nas_choice_prob = {np.around(probs,4)}")
        self.gen_suggested_emb_size(probs, topK=[11, 12, 16, 20], logger=logger)
    
    def gen_suggested_emb_size(self, data, topK=[15, 18, 20, 22], logger=None):
        assert self.num_inputs == data.shape[0]
        rank = np.argsort(np.squeeze(data))
        for K in topK:
            #print(f"Selected_fields:{rank[-K:]}, Droped_fields:{rank[:-K]}.")
            if logger is not None:
                logger.info(f"Top-{K}. Selected_fields:{np.sort(rank[-K:])}, Droped_fields:{np.sort(rank[:-K])}.")
                
class IrazorPretrain(Model):
    def __init__(self, init='xavier', num_inputs=None, input_emb_size_config=[], input_feature_min=[],input_feat_num=[], l2_weight=None, l2_bias=None,
                  target_vec_sizes=[1,2,4,6], temperature=0.50,fid_loss_wt=1e-4,mlp=[], bn=False, ln=False):
        self.l2_weight = l2_weight
        self.l2_bias = l2_bias
        self.num_inputs = num_inputs
        self.search_space = target_vec_sizes
        self.inputs, self.labels, self.training = create_placeholder(num_inputs, tf, True)
        assert num_inputs == len(input_emb_size_config) and num_inputs == len(input_feature_min) and num_inputs==len(input_feat_num)

        all_embs = emb_lookup_multi_emb_size(self.inputs, input_feature_min,input_emb_size_config,input_feat_num)
        print("all_embs:",all_embs)
        self.raw_embedding = tf.concat(all_embs, axis=1)
        
        emb_feature_for_concat, nas_logits, fid_loss = alloc_emb_for_irazor(embed_list=all_embs, 
        target_vec_sizes=target_vec_sizes,temp=temperature,feature_num=input_feat_num, training=self.training)
        self.fid_loss = fid_loss * fid_loss_wt
        all_bias = emb_lookup_multi_emb_size(self.inputs, input_feature_min,[1]*len(input_emb_size_config), input_feat_num)
        print("all_bias", all_bias)
        self.raw_bias = tf.concat(all_bias, axis=1)
        bias_sum = tf.add_n(all_bias) # bs *1
        output, self.layer_kernels, _ = normal_mlp(init="xavier", layer_sizes=mlp, layer_acts=["relu"]*(len(mlp)-1)+["none"], h=emb_feature_for_concat, node_in=int(emb_feature_for_concat.shape[1]), batch_norm=bn, layer_norm=ln, training=self.training)
        print("mlp output", output,"bias_sum",bias_sum) # bs*1
        self.logits = tf.reduce_sum(output+bias_sum, axis=1,keep_dims=False) # bs
        print("lgits", self.logits)
        self.outputs = tf.nn.sigmoid(self.logits)


    def compile(self, loss=None, optimizer1=None, optimizer2=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
                _loss_ = self.loss + self.fid_loss
                #_loss_ = self.loss

                self.l2_loss = get_l2_loss([self.l2_weight, self.l2_bias],
                                               [self.raw_embedding, self.raw_bias])
                if self.l2_loss is not None:
                    _loss_ += self.l2_loss
                all_variable = [v for v in tf.trainable_variables()]
                self.optimizer1 = optimizer1.minimize(loss=_loss_, var_list=all_variable, global_step=global_step)
                self.optimizer2 = None
    def analyse_structure(self, sess, print_full_weight=False, epoch=None, writer=None, logger=None):
        probs = sess.run(["nas_choice_prob:0"])[0]
        print("cur_nas_choice_prob:")
        probs = np.around(probs,4)
        for i in range(len(probs)):
            print(f"feature-{round(i,2)}: {probs[i]}")
        if logger is not None:
            logger.info(f"cur_nas_choice_prob = {np.around(probs,4)}")
        self.gen_suggested_emb_size(probs, cpts=[0.1, 0.3, 0.45, 0.6, 0.7, 0.8], logger=logger)
    
    def gen_suggested_emb_size(self, data, cpts=[0.3,0.4,0.5,0.6], logger=None):
        import pandas as pd
        assert self.num_inputs == data.shape[0]
        size_dims = []
        info = {"field":[]}
        columns = ["field"]
        for i in range(len(self.search_space)):
            dim =self.search_space[i]
            if i==0:
                size_dims.append(0)
            else:
                size_dims.append(self.search_space[i]-self.search_space[i-1])
            info["emb_size_%d_prob"%dim] = []
            columns.append("emb_size_%d_prob"%dim)
        for dim in self.search_space:
            info["emb_size_%d_prob"%dim] = []
            columns.append("emb_size_%d_prob"%dim)
        for cpt in cpts:
            info["cpt_%g"%cpt] = []
            columns.append("cdf_%g" %cpt)
        for i, field  in enumerate(range(self.num_inputs)):
            cur_probs = data[i]
            for j,dim in enumerate(self.search_space):
                info["emb_size_%d_prob"%dim].append(cur_probs[j])
                
            cur_size_dims = [(cur_probs[i], size_dims[i]) for i in range(len(cur_probs))]
            cur_size_dims = sorted(cur_size_dims, key=lambda x: x[0], reverse=True)
            cur_prob = 0.
            cur_size = 0
            percdf_dims = {}
            info["field"].append(field)
            for prob, dim in cur_size_dims:
                cur_prob += prob
                cur_size += dim
                for cpt in cpts:
                    if cur_prob > cpt and str(cpt) not in percdf_dims:
                        percdf_dims[str(cpt)] = cur_size
            for cpt in cpts:
                if str(cpt) not in percdf_dims:
                    info["cpt_%g"%cpt].append(0)
                else:
                    info["cpt_%g"%cpt].append(percdf_dims[str(cpt)])
        df = pd.DataFrame(info)
        cpt_cols = ["cpt_%g"%cpt for cpt in cpts]
        for cpt_col in cpt_cols:
            selected_dims = df[["field",cpt_col]].to_numpy().tolist()
            total_dim = sum(_[1] for _ in selected_dims)
            print(cpt_col," selects total dims = ",total_dim,"selected_dims=", selected_dims)
            if logger is not None:
                logger.info(f"CPT-{cpt_col} selects total dims = {total_dim}, selected_dims = {selected_dims}")

class autofieldPretrain(Model):
    def __init__(self, init='xavier', num_inputs=None, input_emb_size_config=[], input_feature_min=[],input_feat_num=[], l2_weight=None, l2_bias=None,
                  target_vec_sizes=[1,2,4,6], temperature=0.50, mlp=[], bn=False, ln=False):
        self.l2_weight = l2_weight
        self.l2_bias = l2_bias
        self.num_inputs = num_inputs
        self.search_space = target_vec_sizes
        self.inputs, self.labels, self.training = create_placeholder(num_inputs, tf, True)
        assert num_inputs == len(input_emb_size_config) and num_inputs == len(input_feature_min) and num_inputs==len(input_feat_num)

        all_embs = emb_lookup_multi_emb_size(self.inputs, input_feature_min,input_emb_size_config,input_feat_num)
        print("all_embs:",all_embs)
        self.raw_embedding = tf.concat(all_embs, axis=1)
        
        emb_feature_for_concat = alloc_emb_for_autofield(embed_list=all_embs, target_vec_sizes=[0, max(target_vec_sizes)], temp=temperature, training=self.training)
        
        all_bias = emb_lookup_multi_emb_size(self.inputs, input_feature_min,[1]*len(input_emb_size_config), input_feat_num)
        
        self.raw_bias = tf.concat(all_bias, axis=1)
        bias_sum = tf.add_n(all_bias) # bs *1
        output, self.layer_kernels, _ = normal_mlp(init="xavier", layer_sizes=mlp, layer_acts=["relu"]*(len(mlp)-1)+["none"], h=emb_feature_for_concat, node_in=int(emb_feature_for_concat.shape[1]), batch_norm=bn, layer_norm=ln, training=self.training)
        print("mlp output", output,"bias_sum",bias_sum) # bs*1
        self.logits = tf.reduce_sum(output+bias_sum, axis=1,keep_dims=False) # bs
        print("lgits", self.logits)
        self.outputs = tf.nn.sigmoid(self.logits)

    def compile(self, loss=None, optimizer1=None, optimizer2=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
                _loss_ = self.loss
                if not self.l2_weight:
                    self.l2_loss = get_l2_loss([1e-6, 1e-6],
                                               [self.raw_embedding, self.raw_bias])
                    _loss_ += self.l2_loss * 0
                else:
                    self.l2_loss = get_l2_loss([self.l2_weight, self.l2_bias],
                                               [self.raw_embedding, self.raw_bias])
                    _loss_ += self.l2_loss
                all_variable = [v for v in tf.trainable_variables()]
                self.optimizer1 = optimizer1.minimize(loss=_loss_, var_list=all_variable, global_step=global_step)
                self.optimizer2 = None
                
    def analyse_structure(self, sess, print_full_weight=False, epoch=None, writer=None, logger=None):
        probs = sess.run(["nas_choice_prob:0"])[0]
        print("cur_nas_choice_prob:")
        probs = np.around(probs,4)
        data = []
        for i in range(len(probs)):
            print(f"feature-{round(i,2)}: {probs[i]}")
            data.append(probs[i][1])
        if logger is not None:
            logger.info(f"cur_nas_choice_prob = {np.around(probs,4)}")
        self.gen_suggested_emb_size(data, topK=[11, 12, 16, 20], logger=logger)
    
    def gen_suggested_emb_size(self, data, topK=[15, 18, 20, 22], logger=None):
        assert self.num_inputs == len(data)
        rank = np.argsort(data)
        for K in topK:
            #print(f"Top-{K}. Selected_fields:{rank[-K:]}, Droped_fields:{rank[:-K]}.")
            if logger is not None:
                logger.info(f"Top-{K}. Selected_fields:{np.sort(rank[-K:])}, Droped_fields:{np.sort(rank[:-K])}.")
    
class AutoDimPretrain(Model):
    def __init__(self, init='xavier', num_inputs=None, input_emb_size_config=[], input_feature_min=[],input_feat_num=[], l2_weight=None, l2_bias=None,
                  target_vec_sizes=[0,1,2,4,6], global_step=None,mlp=[], bn=False, ln=False):
        self.l2_weight = l2_weight
        self.l2_bias = l2_bias
        self.num_inputs = num_inputs
        self.search_space = target_vec_sizes
        self.inputs, self.labels, self.training = create_placeholder(num_inputs, tf, True)
        assert num_inputs == len(input_emb_size_config) and num_inputs == len(input_feature_min) and num_inputs==len(input_feat_num)

        all_embs = emb_lookup_multi_emb_size(self.inputs, input_feature_min,input_emb_size_config,input_feat_num)
        self.raw_embedding = tf.concat(all_embs, axis=1)
        
        emb_feature_for_concat = alloc_emb_for_autodim(embed_list=all_embs, 
        target_vec_sizes=target_vec_sizes,global_step=global_step,training=self.training )
        all_bias = emb_lookup_multi_emb_size(self.inputs, input_feature_min,[1]*len(input_emb_size_config), input_feat_num)
        print("all_bias", all_bias)
        self.raw_bias = tf.concat(all_bias, axis=1)
        bias_sum = tf.add_n(all_bias) # bs *1
        output, self.layer_kernels, _ = normal_mlp(init="xavier", layer_sizes=mlp, layer_acts=["relu"]*(len(mlp)-1)+["none"], h=emb_feature_for_concat, node_in=int(emb_feature_for_concat.shape[1]), batch_norm=bn, layer_norm=ln, training=self.training)
        print("mlp output", output,"bias_sum",bias_sum) # bs*1
        self.logits = tf.reduce_sum(output+bias_sum, axis=1,keep_dims=False) # bs
        print("lgits", self.logits)
        self.outputs = tf.nn.sigmoid(self.logits)


    def compile(self, loss=None, optimizer1=None, optimizer2=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
                _loss_ = self.loss

                self.l2_loss = get_l2_loss([self.l2_weight, self.l2_bias],
                                               [self.raw_embedding, self.raw_bias])
                if self.l2_loss is not None:
                    _loss_ += self.l2_loss
                all_variable = [v for v in tf.trainable_variables()]
                self.optimizer1 = optimizer1.minimize(loss=_loss_, var_list=all_variable, global_step=global_step)
                self.optimizer2 = None

    def analyse_structure(self, sess, print_full_weight=False, epoch=None, writer=None, logger=None):
        probs = sess.run(["nas_choice_prob:0"])[0]
        print("cur_nas_choice_prob=",  np.around(probs,4))
        if logger is not None:
            logger.info(f"cur_nas_choice_prob = {np.around(probs,4)}")
        argmax_dims = np.argmax(probs, axis=1).tolist()
        field_emb_sizes = []
        for i in range(self.num_inputs):
            field_embed_size = self.search_space[argmax_dims[i]]
            field_emb_sizes.append((i, field_embed_size))
        print("autodim selected sizes=", field_emb_sizes)
        if logger is not None:
            logger.info(f"autodim selected sizes={field_emb_sizes}")
        

class DARTSPretrain(Model):
    def __init__(self, init='xavier', num_inputs=None, input_emb_size_config=[], input_feature_min=[],input_feat_num=[], l2_weight=None, l2_bias=None,
                  target_vec_sizes=[1,2,4,6],mlp=[], bn=False, ln=False):
        self.l2_weight = l2_weight
        self.l2_bias = l2_bias
        self.num_inputs = num_inputs
        self.search_space = target_vec_sizes
        self.inputs, self.labels, self.training = create_placeholder(num_inputs, tf, True)
        assert num_inputs == len(input_emb_size_config) and num_inputs == len(input_feature_min) and num_inputs==len(input_feat_num)

        all_embs = emb_lookup_multi_emb_size(self.inputs, input_feature_min,input_emb_size_config,input_feat_num)
        self.raw_embedding = tf.concat(all_embs, axis=1)
        
        emb_feature_for_concat = alloc_emb_for_darts(embed_list=all_embs, 
        target_vec_sizes=target_vec_sizes)
        all_bias = emb_lookup_multi_emb_size(self.inputs, input_feature_min,[1]*len(input_emb_size_config), input_feat_num)
        print("all_bias", all_bias)
        self.raw_bias = tf.concat(all_bias, axis=1)
        bias_sum = tf.add_n(all_bias) # bs *1
        output, self.layer_kernels, _ = normal_mlp(init="xavier", layer_sizes=mlp, layer_acts=["relu"]*(len(mlp)-1)+["none"], h=emb_feature_for_concat, node_in=int(emb_feature_for_concat.shape[1]), batch_norm=bn, layer_norm=ln, training=self.training)
        print("mlp output", output,"bias_sum",bias_sum) # bs*1
        self.logits = tf.reduce_sum(output+bias_sum, axis=1,keep_dims=False) # bs
        print("lgits", self.logits)
        self.outputs = tf.nn.sigmoid(self.logits)


    def compile(self, loss=None, optimizer1=None, optimizer2=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
                _loss_ = self.loss

                self.l2_loss = get_l2_loss([self.l2_weight, self.l2_bias],
                                               [self.raw_embedding, self.raw_bias])
                if self.l2_loss is not None:
                    _loss_ += self.l2_loss
                all_variable = [v for v in tf.trainable_variables()]
                self.optimizer1 = optimizer1.minimize(loss=_loss_, var_list=all_variable, global_step=global_step)
                self.optimizer2 = None

    def analyse_structure(self, sess, print_full_weight=False, epoch=None, writer=None, logger=None):
        probs = sess.run(["nas_choice_prob:0"])[0]
        print("cur_nas_choice_prob=", probs)
        if logger is not None:
            logger.info(f"cur_nas_choice_prob = {probs}")
        argmax_dims = np.argmax(probs, axis=1).tolist()
        field_emb_sizes = []
        for i in range(self.num_inputs):
            field_embed_size = self.search_space[argmax_dims[i]]
            field_emb_sizes.append((i, field_embed_size))
        print("DARTS selected sizes=", field_emb_sizes)
        if logger is not None:
            logger.info(f"autodim selected sizes={field_emb_sizes}")
        
   
class AutoFIS(Model):
    def __init__(self, init='xavier', num_inputs=None, input_dim=None, embed_size=None, l2_w=None, l2_v=None,
                 layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, norm=False, real_inputs=None,
                 batch_norm=False, layer_norm=False, comb_mask=None, weight_base=0.6, third_prune=False, 
                 comb_mask_third=None, weight_base_third=0.6, retrain_stage=0):
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.l2_ps = l2_v
        self.layer_l2 = layer_l2
        self.retrain_stage = retrain_stage
        self.inputs, self.labels, self.training = create_placeholder(num_inputs, tf, True)
        layer_keeps = drop_out(self.training, layer_keeps)
        inputs, mask, flag, num_inputs = split_data_mask(self.inputs, num_inputs, norm=norm, real_inputs=real_inputs)

        self.xw, xv, _, self.xps = embedding_lookup(init=init, input_dim=input_dim, factor=embed_size, inputs=inputs,
                                            apply_mask=flag, mask=mask, use_b=False, third_order=third_prune)
        self.third_prune = third_prune
        self.xv = xv
        h = tf.reshape(xv, [-1, num_inputs * embed_size])
        h, self.layer_kernels, _ = bin_mlp(init, layer_sizes, layer_acts, layer_keeps, h, num_inputs * embed_size,
                                           batch_norm=batch_norm, layer_norm=layer_norm, training=self.training)
        h = tf.squeeze(h)

        l = linear(self.xw)
        self.cols, self.rows = generate_pairs(range(self.xv.shape[1]),mask=comb_mask)
        t_embedding_matrix = tf.transpose(self.xv, perm=[1, 0, 2])
        left = tf.transpose(tf.gather(t_embedding_matrix, self.rows), perm=[1, 0, 2])
        right = tf.transpose(tf.gather(t_embedding_matrix, self.cols), perm=[1, 0, 2])
        level_2_matrix = tf.reduce_sum(tf.multiply(left, right), axis=-1)
        with tf.variable_scope("edge_weight", reuse=tf.AUTO_REUSE):
            self.edge_weights = tf.get_variable('weights', shape=[len(self.cols)],
                                                initializer=tf.random_uniform_initializer(
                                                minval=weight_base - 0.001,
                                                maxval=weight_base + 0.001))
            normed_wts = tf.identity(self.edge_weights, name="normed_wts")
            tf.add_to_collection("structure", self.edge_weights)
            tf.add_to_collection("edge_weights", self.edge_weights)
            mask = tf.identity(normed_wts, name="unpruned_mask")
            mask = tf.expand_dims(mask, axis=0)
        level_2_matrix = tf.layers.batch_normalization(level_2_matrix, axis=-1, training=self.training,
                                                    reuse=tf.AUTO_REUSE, scale=False, center=False, name='prune_BN')
        level_2_matrix *= mask                                          
        if third_prune:
            self.first, self.second, self.third = generate_pairs(range(self.xps.shape[1]), mask=comb_mask_third, order=3)
            t_embedding_matrix = tf.transpose(self.xps, perm=[1, 0, 2])
            first_embed = tf.transpose(tf.gather(t_embedding_matrix, self.first), perm=[1, 0, 2])
            second_embed = tf.transpose(tf.gather(t_embedding_matrix, self.second), perm=[1, 0, 2])
            third_embed = tf.transpose(tf.gather(t_embedding_matrix, self.third), perm=[1, 0, 2])
            level_3_matrix = tf.reduce_sum(tf.multiply(tf.multiply(first_embed, second_embed), third_embed), axis=-1)
            with tf.variable_scope("third_edge_weight", reuse=tf.AUTO_REUSE):
                self.third_edge_weights = tf.get_variable('third_weights', shape=[len(self.first)],
                                                          initializer=tf.random_uniform_initializer(
                                                              minval=weight_base_third - 0.001,
                                                              maxval=weight_base_third + 0.001))
                third_normed_wts = tf.identity(self.third_edge_weights, name="third_normed_wts")
                tf.add_to_collection("third_structure", self.third_edge_weights)
                tf.add_to_collection("third_edge_weights", self.third_edge_weights)
                third_mask = tf.identity(third_normed_wts, name="third_unpruned_mask")
                third_mask = tf.expand_dims(third_mask, axis=0)
            level_3_matrix = tf.layers.batch_normalization(level_3_matrix, axis=-1, training=self.training,
                                                           reuse=tf.AUTO_REUSE, scale=False, center=False,
                                                           name="level_3_matrix_BN")
            level_3_matrix *= third_mask

        fm_out = tf.reduce_sum(level_2_matrix, axis=-1)
        if third_prune:
            fm_out2 = tf.reduce_sum(level_3_matrix, axis=-1)
        if third_prune:
            self.logits, self.outputs = output([l, fm_out,fm_out2, h, ])
        else:
            self.logits, self.outputs = output([l, fm_out, h, ])

    def analyse_structure(self, sess, print_full_weight=False, epoch=None, writer=None, logger=None):
        import numpy as np
        wts, mask = sess.run(["edge_weight/normed_wts:0", "edge_weight/unpruned_mask:0"])
        if print_full_weight:
            outline = ""
            for j in range(wts.shape[0]):
                outline += str(wts[j]) + ","
            outline += "\n"
            print("log avg auc all weights for(epoch:%s)" % (epoch), outline)
        print("wts", wts[:10])
        print("mask", mask[:10])
        zeros_ = np.zeros_like(mask, dtype=np.float32)
        zeros_[mask == 0] = 1
        print("masked edge_num", sum(zeros_))
        if logger is not None:
            logger.info(f"wts: {wts[:10]}")
            logger.info(f"mask: {mask[:10]}")
            logger.info(f"masked edge_num: {sum(zeros_)}")
        if self.third_prune:
            wts, mask = sess.run(["third_edge_weight/third_normed_wts:0", "third_edge_weight/third_unpruned_mask:0"])
            if print_full_weight:
                outline = ""
                for j in range(wts.shape[0]):
                    outline += str(wts[j]) + ","
                outline += "\n"
                print("third log avg auc all third weights for(epoch:%s)" % (epoch), outline)
                if logger is not None:
                    logger.info(f"third log avg auc all third weights for {epoch}, {outline}")
            print("third wts", wts[:10])
            print("third mask", mask[:10])
            zeros_ = np.zeros_like(mask, dtype=np.float32)
            zeros_[mask == 0] = 1
            print("third masked edge_num", sum(zeros_))
            if logger is not None:
                logger.info(f"third wts: {wts[:10]}")
                logger.info(f"third mask: {mask[:10]}")
                logger.info(f"third masked edge_num: {sum(zeros_)}")

    def compile(self, loss=None, optimizer1=None, optimizer2=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
                _loss_ = self.loss
                if self.third_prune:
                    self.l2_loss = get_l2_loss([self.l2_w, self.l2_v, self.l2_ps, self.layer_l2],
                                               [self.xw, self.xv, self.xps, self.layer_kernels])
                else:
                    self.l2_loss = get_l2_loss([self.l2_w, self.l2_v, self.layer_l2],
                                               [self.xw, self.xv, self.layer_kernels])
                if self.l2_loss is not None:
                    _loss_ += self.l2_loss
                if self.retrain_stage:
                    all_variable = [v for v in tf.trainable_variables()]
                    self.optimizer1 = optimizer1.minimize(loss=_loss_, var_list=all_variable)

                else:
                    all_variable = [v for v in tf.trainable_variables()]
                    if self.third_prune:
                        print("optimizer")
                        weight_second_var = list(set(tf.get_collection("edge_weights")))
                        weight_third_var = list(set(tf.get_collection("third_edge_weights")))
                        weight_var = weight_second_var + weight_third_var
                        weight_var = list(set(weight_var))
                        # weight_var = list(set(tf.get_collection("third_edge_weights")))
                    else:
                        weight_var = list(set(tf.get_collection("edge_weights")))
                    other_var = [i for i in all_variable if i not in weight_var]
                    self.optimizer1 = optimizer1.minimize(loss=_loss_, var_list=other_var)
                    self.optimizer2 = optimizer2.minimize(loss=_loss_, var_list=weight_var)

class PNNPretrainAndRetrain(Model):
    def __init__(self, init='xavier', num_inputs=None, input_emb_size_config=[], input_feature_min=[],input_feat_num=[], l2_weight=None, l2_bias=None,
                  target_vec_sizes=[1,2,4,6], temperature=0.50,fid_loss_wt=1e-4,mlp=[], bn=False, mode="pretrain"):
        assert mode in ["pretrain", "retrain"]
        self.l2_weight = l2_weight
        self.l2_bias = l2_bias
        self.num_inputs = num_inputs
        self.search_space = target_vec_sizes
        self.inputs, self.labels, self.training = create_placeholder(num_inputs, tf, True)
        assert num_inputs == len(input_emb_size_config) and num_inputs == len(input_feature_min) and num_inputs==len(input_feat_num)

        all_embs = emb_lookup_multi_emb_size(self.inputs, input_feature_min,input_emb_size_config,input_feat_num)
        self.raw_embedding = tf.concat(all_embs, axis=1)
        self.losses = []
        self.mlp=mlp
        self.mlp_bn = bn
        self.mode = mode
        if mode == "pretrain":
            emb_feature_for_concat, nas_logits, fid_loss = alloc_emb_for_irazor(embed_list=all_embs, 
            target_vec_sizes=target_vec_sizes,temp=temperature,feature_num=input_feat_num,training=self.training)
            self.interaction_base_embs = tf.split(emb_feature_for_concat, num_or_size_splits=len(all_embs), axis=1)
            self.fid_loss = fid_loss* fid_loss_wt
        else: #perform padding
            padding_size = max(input_emb_size_config)
            self.interaction_base_embs = []
            for i,raw_emb in enumerate(all_embs):
                raw_size = int(raw_emb.shape[1])
                if raw_size < padding_size:
                    padding_matrix = get_variable(name='{}_transform_{}_to_{}'.format(i,raw_size, padding_size), \
                                            shape=[raw_size,padding_size])
                    padding_bias = get_variable(name='{}_transform_{}_to_{}_wt'.format(i,raw_size, padding_size), \
                                            shape=[1,padding_size])
                    cur_emb = tf.matmul(raw_emb, padding_matrix) + padding_bias # bs* emb_raw matmul  emb_raw * paddingsize -> bs*padding_size + bias
                    self.interaction_base_embs.append(cur_emb)
                else:
                    self.interaction_base_embs.append(raw_emb)
            self.fid_loss=None

        all_bias = emb_lookup_multi_emb_size(self.inputs, input_feature_min,[1 if ele>0 else 0 for ele in input_emb_size_config], input_feat_num)
        print("all_bias", all_bias)
        self.raw_bias = tf.concat(all_bias, axis=1)
        self.build_structure()
    
    def build_structure(self):
        left_emb = []
        right_emb = []
        for i in range(len(self.interaction_base_embs)):
            for j in range(i+1, len(self.interaction_base_embs)):
                left_emb.append(self.interaction_base_embs[i])
                right_emb.append(self.interaction_base_embs[j])
        left_full = tf.stack(left_emb, axis=1) # bs* comb * emb_szize
        right_full = tf.stack(right_emb, axis=1) # bs* comb * emb_size
        pnn_output = tf.reduce_sum(left_full * right_full, axis=2)
        
        bias_sum = tf.reduce_sum(self.raw_bias, axis=1, keep_dims=True) # bs *1
        nn_concat = tf.concat([self.raw_embedding, pnn_output], axis=1)  
        output, self.layer_kernels, _ = normal_mlp(init="xavier", layer_sizes=self.mlp,\
                        layer_acts=["relu"]*(len(self.mlp)-1)+["none"], h=nn_concat, 
                        node_in=int(nn_concat.shape[1]), 
                        batch_norm=self.mlp_bn, layer_norm=None, training=self.training)
        print("mlp output", output,"bias_sum",bias_sum) # bs*1
        self.logits = tf.reduce_sum(output+bias_sum, axis=1,keep_dims=False) # bs
        print("logits", self.logits)
        self.outputs = tf.nn.sigmoid(self.logits)


    def compile(self, loss=None, optimizer1=None, optimizer2=None, global_step=None, pos_weight=1.0):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(loss(logits=self.logits, targets=self.labels, pos_weight=pos_weight))
                _loss_ = self.loss 
                if self.fid_loss is not None:
                    _loss_ += self.fid_loss
                self.l2_loss = get_l2_loss([self.l2_weight, self.l2_bias],
                                               [self.raw_embedding, self.raw_bias])
                if self.l2_loss is not None:
                    _loss_ += self.l2_loss
                all_variable = [v for v in tf.trainable_variables()]
                self.optimizer1 = optimizer1.minimize(loss=_loss_, var_list=all_variable, global_step=global_step)
                self.optimizer2 = None
    
    def analyse_structure(self, sess, print_full_weight=False, epoch=None, writer=None, logger=None):
        if self.mode == "pretrain":
            probs = sess.run(["nas_choice_prob:0"])[0]
            #print("cur_nas_choice_prob=", probs)
            probs = np.around(probs,4)
            for i in range(len(probs)):
                print(f"feature-{round(i,2)}: {probs[i]}")
            if logger is not None:
                logger.info(f"cur_nas_choice_prob = {probs}")
            self.gen_suggested_emb_size(data=probs, cpts=[0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], logger=logger)
        else:
            pass
    
    def gen_suggested_emb_size(self, data, cpts=[0.3,0.4,0.5,0.6], logger=None):
        import pandas as pd
        assert self.num_inputs == data.shape[0]
        size_dims = []
        info = {"field":[]}
        columns = ["field"]
        for i in range(len(self.search_space)):
            dim =self.search_space[i]
            if i==0:
                size_dims.append(0)
            else:
                size_dims.append(self.search_space[i]-self.search_space[i-1])
            info["emb_size_%d_prob"%dim] = []
            columns.append("emb_size_%d_prob"%dim)
        for dim in self.search_space:
            info["emb_size_%d_prob"%dim] = []
            columns.append("emb_size_%d_prob"%dim)
        for cpt in cpts:
            info["cpt_%g"%cpt] = []
            columns.append("cdf_%g" %cpt)
        for i, field  in enumerate(range(self.num_inputs)):
            cur_probs = data[i]
            for j,dim in enumerate(self.search_space):
                info["emb_size_%d_prob"%dim].append(cur_probs[j])
                
            cur_size_dims = [(cur_probs[i], size_dims[i]) for i in range(len(cur_probs))]
            cur_size_dims = sorted(cur_size_dims, key=lambda x: x[0], reverse=True)
            cur_prob = 0.
            cur_size = 0
            percdf_dims = {}
            info["field"].append(field)
            for prob, dim in cur_size_dims:
                cur_prob += prob
                cur_size += dim
                for cpt in cpts:
                    if cur_prob > cpt and str(cpt) not in percdf_dims:
                        percdf_dims[str(cpt)] = cur_size
            for cpt in cpts:
                if str(cpt) not in percdf_dims:
                    info["cpt_%g"%cpt].append(0)
                else:
                    info["cpt_%g"%cpt].append(percdf_dims[str(cpt)])
        df = pd.DataFrame(info)
        cpt_cols = ["cpt_%g"%cpt for cpt in cpts]
        for cpt_col in cpt_cols:
            selected_dims = df[["field",cpt_col]].to_numpy().tolist()
            total_dim = sum(_[1] for _ in selected_dims)
            print(cpt_col," selects total dims = ",total_dim,"selected_dims=", selected_dims)
            if logger is not None:
                logger.info(f"CPT-{cpt_col} selects total dims = {total_dim}, selected_dims = {selected_dims}")

class DeepFMPretrainAndRetrain(PNNPretrainAndRetrain):
    def build_structure(self):
        left_emb = []
        right_emb = []
        for i in range(len(self.interaction_base_embs)):
            for j in range(i+1, len(self.interaction_base_embs)):
                left_emb.append(self.interaction_base_embs[i])
                right_emb.append(self.interaction_base_embs[j])
        left_full = tf.stack(left_emb, axis=1) # bs* comb * emb_szize
        right_full = tf.stack(right_emb, axis=1) # bs* comb * emb_size
        fm_output = tf.reduce_sum(tf.reduce_sum(left_full * right_full, axis=2), axis=1, keep_dims=True)  # bs* 1
        bias_sum = tf.reduce_sum(self.raw_bias, axis=1, keep_dims=True) # bs *1
        nn_concat = tf.concat([self.raw_embedding], axis=1)  
        output, self.layer_kernels, _ = normal_mlp(init="xavier", layer_sizes=self.mlp,\
                        layer_acts=["relu"]*(len(self.mlp)-1)+["none"], h=nn_concat, 
                        node_in=int(nn_concat.shape[1]), 
                        batch_norm=self.mlp_bn, layer_norm=None, training=self.training)
        print("mlp output", output,"bias_sum",bias_sum) # bs*1
        self.logits = tf.reduce_sum(output+bias_sum+fm_output, axis=1,keep_dims=False) # bs
        print("lgits", self.logits)
        self.outputs = tf.nn.sigmoid(self.logits)


class xDeepFMPretrainAndRetrain(PNNPretrainAndRetrain):
    def __init__(self, init='xavier', num_inputs=None, input_emb_size_config=[], input_feature_min=[],input_feat_num=[], l2_weight=None, l2_bias=None,
                  target_vec_sizes=[1,2,4,6], temperature=0.50,fid_loss_wt=1e-4, mlp=[], bn=False, mode="pretrain", cin_layers=[32, 32, 32]):
        #super().__init_subclass__()
        self.cin_layers=cin_layers #代表每层交互的field数量
        super(xDeepFMPretrainAndRetrain, self).__init__(init, num_inputs, input_emb_size_config, input_feature_min,input_feat_num, l2_weight, l2_bias,
                  target_vec_sizes, temperature,fid_loss_wt, mlp, bn, mode) 
        
   
  
    def build_structure(self):
        X_0 = tf.stack(self.interaction_base_embs, axis=1) # bs* num1 * emb_size
        X_i = X_0
        raw_emb_size = X_0.shape[2]
        cin_outputs = []
        for i in range(len(self.cin_layers)):
            hadamard_tensor = tf.einsum("bhd,bmd->bhmd", X_0, X_i) # bs * num1 * numx * emb_size
            #cin个数是每个高阶的生成的field数量
            last_num = int(X_i.shape[1])
            raw_num = int(X_0.shape[1])
            cur_layer_size = self.cin_layers[i]
            tmp_tensor = tf.reshape(hadamard_tensor, [-1, last_num*raw_num, raw_emb_size]) # bs* (num1*numx) * emb
            compress_wt = get_variable(name='cin_{}_wt_{}_new_{}'.format(i, last_num,cur_layer_size), \
                                            shape=[last_num*raw_num, cur_layer_size])
            compress_bias = get_variable(name='cin_{}_bias_{}_new_{}'.format(i, last_num,cur_layer_size), \
                                            shape=[1, cur_layer_size,1])
            # bs*(dim1) * emb     dim1 * dim2
            compressed_output = tf.einsum("bed, ef-> bfd", tmp_tensor, compress_wt)# bs* layer * emb 
            X_i = compressed_output + compress_bias
            cin_outputs.append(tf.reduce_sum(X_i, axis=2)) # bs* layer
        cin_final_output_concat = tf.concat(cin_outputs, axis=1) # bs* sum(cin_layers)
        in_size = sum(self.cin_layers)
        wt = get_variable(name='cin_pooling_in_{}_out1'.format(in_size), \
                                        shape=[in_size, 1])
        cin_final_res = tf.matmul(cin_final_output_concat, wt) # bs * 1
        bias_sum = tf.reduce_sum(self.raw_bias, axis=1, keep_dims=True) # bs *1
        nn_concat = tf.concat([self.raw_embedding], axis=1)  
        output, self.layer_kernels, _ = normal_mlp(init="xavier", layer_sizes=self.mlp,\
                        layer_acts=["relu"]*(len(self.mlp)-1)+["none"], h=nn_concat, 
                        node_in=int(nn_concat.shape[1]), 
                        batch_norm=self.mlp_bn, layer_norm=None, training=self.training)
        print("mlp output", output,"bias_sum",bias_sum) # bs*1
        self.logits = tf.reduce_sum(output+bias_sum+cin_final_res, axis=1,keep_dims=False) # bs
        print("lgits", self.logits)
        self.outputs = tf.nn.sigmoid(self.logits)