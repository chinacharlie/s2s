import tensorflow as tf
from hyperparams import Hyperparams as hp
from gan import *
import time
from  data_load import NetDataFlow
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"]=""   

def main():

    train_mod = "mel" 
    if len(sys.argv) > 1:
        train_mod = sys.argv[1]

    g = GDNet(True, 0.9)
    da_optim = tf.train.AdamOptimizer(hp.lr).minimize(g.daloss(), var_list=g.da_vars)
    db_optim = tf.train.AdamOptimizer(hp.lr).minimize(g.dbloss(), var_list=g.db_vars)
    d_optim = tf.train.AdamOptimizer(hp.lr).minimize(g.dloss(), var_list=g.d_vars)

    ga_optim = tf.train.AdamOptimizer(hp.lr).minimize(g.galoss(), var_list=g.g_vars)
    gb_optim = tf.train.AdamOptimizer(hp.lr).minimize(g.gbloss(), var_list=g.g_vars) 
    g_optim = tf.train.AdamOptimizer(hp.lr).minimize(g.gloss(), var_list=g.g_vars)

    decode_optim = tf.train.AdamOptimizer(hp.lr).minimize(g.decodeloss(), var_list=g.decode_vars)


    sum_op = tf.summary.merge([g.ga_loss_sum, g.gb_loss_sum, g.da_loss_sum, g.db_loss_sum])

    init_op = tf.global_variables_initializer()
    g.sess.run(init_op)
 
    reload_ok, start_epoch = g.load()

    d = NetDataFlow()
    counter = 1
    start_time = time.time()
    number_pre_epoch = 10
    for epoch in range(start_epoch + 1, 100000):

        #g.save(hp.mode_dir, epoch) 
        for i in range(number_pre_epoch):
            a_specs, b_specs, a_mels, b_mels, _, _ = d.getbatch()
 
            summary_str, ga_loss, gb_loss, g_cycle_loss, da_loss, db_loss = g.sess.run( \
                        [sum_op , g.galoss(), g.gbloss(), g.cycleloss(), g.daloss(),g.dbloss()], \
                        feed_dict={ g.x_a_mel: a_mels, g.x_b_mel : b_mels })

            g.writer.add_summary(summary_str, epoch * number_pre_epoch + i)
            
            loss_info = "[%2d][%2d/100]%05.0f, ga_loss:%03.2f, gb_loss:%03.2f, c_loss:%03.2f, da_loss:%02.2f, db_loss:%02.2f" \
                        % (epoch, i, time.time() - start_time, ga_loss, gb_loss, g_cycle_loss, da_loss, db_loss)
            
            gb_id_loss, gb_d_loss, gb_c_loss, gb_w_loss, gb_x_loss = g.sess.run( \
                        g.gbloss_ex(), \
                        feed_dict={ g.x_a_mel: a_mels, g.x_b_mel : b_mels})

            loss_info += ' gb_id:%03.2f gb_d:%03.2f gb_c:%03.2f gb_w:%03.2f' % (gb_id_loss, gb_d_loss, gb_c_loss, gb_w_loss)
            
            loss_info += ' d:%.2f id:%.2f w:%.2f' % (hp.loss_lambda_d, hp.loss_lambda_id, hp.loss_lambda_w)
            print(loss_info)



            
            for j in range(10):
                _, _, a_mels, b_mels, _, _ = d.getbatch()
                g.sess.run([d_optim], feed_dict={ g.x_a_mel: a_mels, g.x_b_mel : b_mels })
                g.sess.run([g_optim], feed_dict={ g.x_a_mel: a_mels, g.x_b_mel : b_mels })
            
            '''
            for j in range(10):
                a_specs, b_specs, a_mels, b_mels, _, _ = d.getbatch()
                g.sess.run([decode_optim], feed_dict={ g.x_a_mel: a_mels, g.x_b_mel : b_mels, g.x_a_spec: a_specs, g.x_b_spec: b_specs })
            '''


         

        print('save mode %d' % epoch)
        g.save(hp.mode_dir, epoch)


if __name__ == "__main__":
    main()


