import tensorflow as tf
from hyperparams import Hyperparams as hp
from net import *
import time
from time import strftime, localtime
from  data_load import NetDataFlow
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"   

def main():
    g = S2SNet(True)
    f_optim = tf.train.AdamOptimizer(hp.lr).minimize(g.floss(), var_list=g.fvars)
    g_optim = tf.train.AdamOptimizer(hp.lr).minimize(g.gloss(), var_list=g.gvars)
    loop_optim = tf.train.AdamOptimizer(hp.lr).minimize(g.looploss(), var_list=g.gvars)
    
    
   

    sum_op = tf.summary.merge([g.f_loss_sum, g.g_loss_sum])

    init_op = tf.global_variables_initializer()
    g.sess.run(init_op)
 
    reload_ok, start_epoch = g.loadg()

    d1 = NetDataFlow('../data_thchs30/prestf', True)
    d2 = NetDataFlow('../data_thchs30/data')

    counter = 1
    start_time = time.time()
    number_pre_epoch = 10
    for epoch in range(start_epoch + 1, 100000):

        #g.save(hp.mode_dir, epoch) 
        for i in range(number_pre_epoch):
            x_mels, x_specs, x_labels = d1.get_data()
 
            summary_str, g_loss = g.sess.run( \
                        [sum_op , g.gloss()], \
                        feed_dict={ g.x_mel: x_mels, g.x_spec : x_specs, g.x_label : x_labels})

            x_mels, x_specs, x_labels = d2.get_data()
 
            loop_loss = g.sess.run( \
                        g.looploss(), \
                        feed_dict={ g.x_mel: x_mels, g.x_spec : x_specs, g.x_label : x_labels})

            g.writer.add_summary(summary_str, epoch * number_pre_epoch + i)
            
            loss_info = "%s %s [%2d][%2d/100]%05.0f, g_loss:%03.5f, loop_loss:%03.5f" % (os.getcwd(), strftime("%Y-%m-%d %H:%M:%S"), epoch, i, time.time() - start_time, g_loss, loop_loss)
        
            print(loss_info)

            for j in range(10):
                x_mels, x_specs, x_labels = d1.get_data()
                g.sess.run([g_optim], feed_dict={ g.x_mel: x_mels, g.x_spec : x_specs, g.x_label : x_labels })
                
                #x_mels, x_specs, x_labels = d2.get_data()
                #g.sess.run([loop_optim], feed_dict={ g.x_mel: x_mels, g.x_spec : x_specs, g.x_label : x_labels })



        print('save mode %d' % epoch)
        g.saveg(hp.g_mode_dir, epoch)


if __name__ == "__main__":
    main()


