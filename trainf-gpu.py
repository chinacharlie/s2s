

import tensorflow as tf
from hyperparams import Hyperparams as hp
from net import S2SNet
import time
from time import strftime, localtime
from  data_load import NetDataFlow
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"]=""   

def main():
    g = S2SNet(True)
    
   

    sum_op = tf.summary.merge([g.f_loss_sum])

    init_op = tf.global_variables_initializer()
    g.sess.run(init_op)
 
    reload_ok, start_epoch = g.load()

    d1 = NetDataFlow('../data_thchs30/data')
    d2 = NetDataFlow('../data_thchs30/data-cafe')
    d3 = NetDataFlow('../data_thchs30/data-car')
    
    counter = 1
    start_time = time.time()
    number_pre_epoch = 1
    for epoch in range(start_epoch + 1, 100000):

        #g.save(hp.mode_dir, epoch) 
        for i in range(number_pre_epoch):
            x_mels, x_specs, x_labels = d1.get_data()
 
            summary_str, f_loss = g.sess.run( \
                        [sum_op , g.f_loss], \
                        feed_dict={ g.x_mel: x_mels, g.x_spec : x_specs, g.x_label : x_labels})

            g.writer.add_summary(summary_str, epoch * number_pre_epoch + i)
            
            loss_info = "%s [%2d][%2d/100]%05.0f, f_loss:%03.2f" % (strftime("%Y-%m-%d %H:%M:%S", localtime()),epoch, i, time.time() - start_time, f_loss)
        
            print(loss_info)

            for j in range(1):
                x_mels, x_specs, x_labels = d1.get_data(hp.batch_size * hp.gpu_num)
                g.sess.run([g.f_train_op], feed_dict={ g.x_mel: x_mels, g.x_spec : x_specs, g.x_label : x_labels })
                x_mels, x_specs, x_labels = d2.get_data(hp.batch_size * hp.gpu_num)
                g.sess.run([g.f_train_op], feed_dict={ g.x_mel: x_mels, g.x_spec : x_specs, g.x_label : x_labels })
                x_mels, x_specs, x_labels = d3.get_data(hp.batch_size * hp.gpu_num)
                g.sess.run([g.f_train_op], feed_dict={ g.x_mel: x_mels, g.x_spec : x_specs, g.x_label : x_labels })

        print('save mode %d' % epoch)
        g.save(hp.mode_dir, epoch)


if __name__ == "__main__":
    main()


