import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import librosa
from modules import prenet, cbhg, normalize, GRULayer
import numpy as np
import os

from hyperparams import Hyperparams as hp



def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    
    return average_grads


# By default, all variables will be placed on '/gpu:0'
# So we need a custom device function, to assign all variables to '/cpu:0'
# Note: If GPUs are peered, '/gpu:0' can be a faster option
PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']

def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign

def floss(x_mel, x_label, logits):
        indices = tf.where(tf.not_equal(tf.cast(x_label, tf.float32), 0.))
        target = tf.SparseTensor(indices=indices, values=tf.cast(tf.gather_nd(x_label, indices), tf.int32), dense_shape=tf.cast(tf.shape(x_label), tf.int64))
        sequence_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(x_mel, reduction_indices=2), 0.), tf.int32), reduction_indices=1)
        loss_ = tf.nn.ctc_loss(target, logits, sequence_len, time_major=False)
        loss_ = tf.reduce_mean(loss_)
        return loss_

def gloss(x_spec, x_mel, g_spec, g_mel):
        loss_spec = tf.reduce_mean(tf.squared_difference(g_spec , x_spec))
        loss_mel = tf.reduce_mean(tf.squared_difference(g_mel , x_mel))
        return loss_mel + loss_spec

class S2SNet:
    def __init__(self, is_training = True):
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu)
        self.sess = tf.Session() #config=tf.ConfigProto(gpu_options=gpu_options)) 
        self.is_training = is_training
        self.build_graph()

    def get_inputs(self):
        x_mel = tf.placeholder(tf.float32, (None, None, hp.n_mels), name = 'x_mel')
        x_spec = tf.placeholder(tf.float32, (None, None, hp.n_fft // 2 + 1), name = 'x_spec')
        x_label = tf.placeholder(tf.float32, (None, None), name = 'x_label')
        return x_mel, x_spec, x_label
 
    def gnet(self, feature, is_training=True, reuse=None):

        #print(self.x_mel.shape[-1])
        prenet_out = tf.layers.dense(feature, self.x_mel.shape[-1], name='prenet_out_dense', reuse=reuse)

        prenet_out = prenet(prenet_out,
                            num_units=[hp.hidden_units, hp.hidden_units],
                            dropout_rate=hp.dropout_rate,
                            is_training=is_training,
                            reuse=reuse)  # (N, T, E/2)
        
        # CBHG1: mel-scale
        pred_mel, _ = cbhg(prenet_out, hp.num_banks, hp.hidden_units,
                        hp.num_highway_blocks, hp.norm_type, is_training,
                        scope="cbhg_gnet_mel",
                        reuse=reuse)

        g_mel = tf.layers.dense(pred_mel, self.x_mel.shape[-1], name='g_mel', reuse=reuse)  # (N, T, n_mel)
        print(g_mel)
        pred_spec = tf.layers.dense(g_mel, hp.hidden_units, name='pred_spec_dense', reuse=reuse)  # (N, T, n_mels)

        pred_spec, _ = cbhg(pred_spec, hp.num_banks, hp.hidden_units,
                   hp.num_highway_blocks, hp.norm_type, is_training, 
                   scope="cbhg_gnet_spec",
                   reuse=reuse)

        g_spec = tf.layers.dense(pred_spec, self.x_spec.shape[-1], name = 'g_spec', reuse=reuse)
        return g_spec, g_mel

 
    
 
    def fnet(self, mel, is_training=True, reuse=None):
        logits = tf.layers.dense(mel, hp.len_chinese_ppgs, trainable=is_training, name='fnet_logits_dense', reuse=reuse)
        return logits

        prenet_out = prenet(mel,
                            num_units=[hp.hidden_units, hp.hidden_units // 2],
                            dropout_rate=hp.dropout_rate,
                            is_training=is_training,
                            reuse=reuse)  # (N, T, E/2)
        # CBHG1: mel-scale
        out, _ = cbhg(prenet_out, hp.num_banks, hp.hidden_units // 2,
                        hp.num_highway_blocks, hp.norm_type, is_training,
                        scope="fnet_cbhg",
                        reuse=reuse)

        #out = LstmLayer(prenet_out, hp.train1.hidden_units, is_training)

        # Final linear projection
        logits = tf.layers.dense(out, hp.len_chinese_ppgs, trainable=is_training, name='fnet_logits_dense', reuse=reuse)  # (N, T, V)
        ppgs = tf.nn.softmax(logits / hp.t, name='ppgs')  # (N, T, V)
        preds = tf.to_int32(tf.argmax(logits, axis=-1))  # (N, T)
        
        decoded = tf.transpose(logits, perm=[1, 0, 2])    
        sequence_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(self.x_mel, reduction_indices=2), 0.), tf.int32), reduction_indices=1)  
        decoded, _ = tf.nn.ctc_beam_search_decoder(decoded, sequence_len, merge_repeated=False)    
        decoded = tf.sparse_to_dense(decoded[0].indices,decoded[0].dense_shape,decoded[0].values)
        return logits, ppgs, preds, decoded

    def floss(self):
        indices = tf.where(tf.not_equal(tf.cast(self.x_label, tf.float32), 0.))    
        target = tf.SparseTensor(indices=indices, values=tf.cast(tf.gather_nd(self.x_label, indices), tf.int32), dense_shape=tf.cast(tf.shape(self.x_label), tf.int64))      
        sequence_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(self.x_mel, reduction_indices=2), 0.), tf.int32), reduction_indices=1)    
        loss_ = tf.nn.ctc_loss(target, self.logits, sequence_len, time_major=False)  
        loss_ = tf.reduce_mean(loss_)
        return loss_
    
    def looploss(self):

        indices = tf.where(tf.not_equal(tf.cast(self.x_label, tf.float32), 0.))    
        target = tf.SparseTensor(indices=indices, values=tf.cast(tf.gather_nd(self.x_label, indices), tf.int32), dense_shape=tf.cast(tf.shape(self.x_label), tf.int64))      
        sequence_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(self.x_mel, reduction_indices=2), 0.), tf.int32), reduction_indices=1)    
        loss_ = tf.nn.ctc_loss(target, self.loop_logits, sequence_len, time_major=False)  
        loss_ = tf.reduce_mean(loss_) * 0.001 #考虑到和gloss的巨大差异
        return loss_

    #def dloos(self):
    #    loss_ = 0.95 * tf.log(self.dx) + 0.05 * tf.log(1 - self.dx)
    #    loss_ = loss_ + 0.95 * tf.log(1 - self.dg) + 0.05 * tf.log(self.dg)
    #    return loss_

    def gloss(self):
        loss_spec = tf.reduce_mean(tf.squared_difference(self.g_spec , self.x_spec))
        loss_mel = tf.reduce_mean(tf.squared_difference(self.g_mel , self.x_mel))
        return loss_mel + loss_spec

    def create_net(self):
        
        with tf.device('/cpu:0'):
            f_tower_grads = []
            f_reuse_vars = False
            g_tower_grads = []
            g_reuse_vars = False
            for i in range(hp.gpu_num):
                with tf.device(assign_to_device('/cpu:{}'.format(i), ps_device='/cpu:0')):
                #with tf.device('cpu:0'):
                    print("gpu_num %d" % i)
                    x_mel = self.x_mel[i * hp.batch_size: (i+1) * hp.batch_size]
                    x_label = self.x_label[i * hp.batch_size: (i+1) * hp.batch_size]
                    x_spec = self.x_spec[i * hp.batch_size: (i+1) * hp.batch_size] 
                    with tf.variable_scope('fnet'):
                        logits = self.fnet(x_mel, reuse=f_reuse_vars)
                    if f_reuse_vars == False:
                        self.logits = logits 
                    else:
                        self.logits = tf.concat((self.logits, logits), 0)
                 
                        
                    self.f_loss = floss(x_mel, x_label, logits)
                    t_vars = tf.trainable_variables()
                    f_vars = [var for var in t_vars if 'fnet' in var.name]
                    f_optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
                    f_grads = f_optimizer.compute_gradients(self.f_loss, var_list=f_vars)
                    self.f_vars = f_vars
                    
                 


                f_reuse_vars = True
                f_tower_grads.append(f_grads)
              

            f_tower_grads = average_gradients(f_tower_grads)
            self.f_train_op = f_optimizer.apply_gradients(f_tower_grads)   
       



    def build_graph(self):
        
        self.x_mel, self.x_spec, self.x_label = self.get_inputs()
        self.create_net()
     

        self.f_loss_sum = tf.summary.scalar("f_loss_sum", self.f_loss)        
      

        self.saver = tf.train.Saver(max_to_keep=3)

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

 
        
    
    def save(self, checkpoint_dir=hp.mode_dir, epoch = 0):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, "f"),
                        global_step=epoch)


    def load(self, checkpoint_dir = hp.mode_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            
            epoch = int (ckpt_name.split('-')[1])
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True, epoch
        else:
            return False, -1
 

if __name__ == '__main__':
    n=S2SNet()
    #n2.build_graph()
    n.load()
    n.save()
