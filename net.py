import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import librosa
from modules import prenet, cbhg, normalize, GRULayer
import numpy as np
import os

from hyperparams import Hyperparams as hp


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

        prenet_out = tf.layers.dense(feature, hp.hidden_units, reuse=reuse)

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

        pred_spec = tf.layers.dense(g_mel, hp.hidden_units, reuse=reuse)  # (N, T, n_mels)

        pred_spec, _ = cbhg(pred_spec, hp.num_banks, hp.hidden_units,
                   hp.num_highway_blocks, hp.norm_type, is_training, 
                   scope="cbhg_gnet_spec",
                   reuse=reuse)

        g_spec = tf.layers.dense(pred_spec, self.x_spec.shape[-1], name = 'g_spec', reuse=reuse)


        return g_spec, g_mel

    def fnet(self, mel, is_training=True, reuse=None):
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
        logits = tf.layers.dense(out, hp.len_chinese_ppgs, trainable=is_training, reuse=reuse)  # (N, T, V)
        ppgs = tf.nn.softmax(logits / hp.t, name='ppgs')  # (N, T, V)
        preds = tf.to_int32(tf.argmax(logits, axis=-1))  # (N, T)

        decoded = tf.transpose(logits, perm=[1, 0, 2])    
        sequence_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(mel, reduction_indices=2), 0.), tf.int32), reduction_indices=1)  
        decoded, _ = tf.nn.ctc_beam_search_decoder(decoded, sequence_len, merge_repeated=False)    
        decoded = tf.sparse_to_dense(decoded[0].indices,decoded[0].dense_shape,decoded[0].values)

        return out, logits, ppgs, preds, decoded

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
        loss_ = tf.reduce_mean(loss_) * 0.0001 #考虑到和gloss的巨大差异
        return loss_

    #def dloos(self):
    #    loss_ = 0.95 * tf.log(self.dx) + 0.05 * tf.log(1 - self.dx)
    #    loss_ = loss_ + 0.95 * tf.log(1 - self.dg) + 0.05 * tf.log(self.dg)
    #    return loss_

    def gloss(self):
        loss_spec = tf.reduce_mean(tf.squared_difference(self.g_spec , self.x_spec))
        loss_mel = tf.reduce_mean(tf.squared_difference(self.g_mel , self.x_mel))
        return loss_mel + loss_spec


    def build_graph(self, net_name='g'):
        self.x_mel, self.x_spec, self.x_label = self.get_inputs()

        with tf.variable_scope('fnet'):
            self.mid, self.logits, self.ppgs, self.preds, self.decoded = self.fnet(self.x_mel)

        with tf.variable_scope('gnet'):
            self.g_spec, self.g_mel = self.gnet(self.logits)

        with tf.variable_scope('fnet'):
            self.loop_mid, self.loop_logits, self.loop_ppgs, self.loop_preds, self.loop_decoded = self.fnet(self.g_mel, reuse=True)
 
     
        self.t_vars = tf.trainable_variables()
        self.fvars = [var for var in self.t_vars if 'fnet' in var.name]
        self.gvars = [var for var in self.t_vars if 'gnet' in var.name]



        self.f_loss_sum = tf.summary.scalar("f_loss_sum", self.floss())
        self.g_loss_sum = tf.summary.scalar("g_loss_sum", self.gloss())
        
        
        self.f_saver = tf.train.Saver(max_to_keep=3, var_list = self.fvars)
        self.g_saver = tf.train.Saver(max_to_keep=3, var_list = self.t_vars)

        
        
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)


    def info(self):
        x_mean = tf.reduce_mean(self.x_mel)
        g_mean = tf.reduce_mean(self.g_mel)
        x_max = tf.reduce_max(self.x_mel)
        g_max = tf.reduce_max(self.g_mel)
        return x_mean, g_mean, x_max, g_max

    def savef(self, checkpoint_dir=hp.f_mode_dir, epoch = 0):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.f_saver.save(self.sess,
                        os.path.join(checkpoint_dir, "fnet"),
                        global_step=epoch)
        
    
    def saveg(self, checkpoint_dir=hp.g_mode_dir, epoch = 0):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.g_saver.save(self.sess,
                        os.path.join(checkpoint_dir, "gnet"),
                        global_step=epoch)

    def loadf(self, checkpoint_dir=hp.f_mode_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print("load %s" % (ckpt_name))
            epoch = int (ckpt_name.split('-')[1])
            self.f_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True, epoch
        else:
            return False, -1


    def loadg(self, checkpoint_dir=hp.g_mode_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print("load %s" % (ckpt_name))
            epoch = int (ckpt_name.split('-')[1])
            self.g_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True, epoch
        else:
            ckpt = tf.train.get_checkpoint_state(hp.f_mode_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                print("load %s" % (ckpt_name))
                epoch = 1
                self.f_saver.restore(self.sess, os.path.join(hp.f_mode_dir, ckpt_name))
                return True, epoch
            else:
                return False, -1

if __name__ == '__main__':
    n2=S2SNet()
    #n2.build_graph()
