import tensorflow as tf
from hyperparams import Hyperparams as hp
from net import *
import time
from  data_load import *
from utils import normalize_0_1, denormalize_0_1
from audio import spec2wav, inv_preemphasis, db2amp, denormalize_db, amp2db
import os
from net import *

os.environ["CUDA_VISIBLE_DEVICES"]=""   

def sumspecimage(spec, spec_name):
    spec = denormalize_db(spec, hp.max_db, hp.min_db)
    spec = db2amp(spec)

    spec_image = spec.transpose(0,2,1)
    heatmap = np.expand_dims(spec_image, 3)
    tf.summary.image(spec_name, heatmap, max_outputs=spec_image.shape[0])

    out_spec = np.power(np.maximum(spec, 0), 1) #hp.emphasis_magnitude)
    out_audio = np.array(list(map(lambda spec: spec2wav(spec.T, hp.n_fft, hp.win_length, hp.hop_length,
                                                 hp.n_iter), out_spec)))
    
    out_audio = inv_preemphasis(out_audio, coeff=hp.preemphasis)
    tf.summary.audio(spec_name, out_audio, hp.sr, max_outputs=hp.batch_size) 



def sumimage(mel, mel_name):
    mel = mel #+ 0.001 * np.random.standard_normal([hp.batch_size, hp.duration * hp.n_mels, hp.n_mels])
    mel_image = mel.transpose(0,2,1)
    heatmap = np.expand_dims(mel_image, 3)
    tf.summary.image(mel_name, heatmap, max_outputs=mel_image.shape[0])


    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)
    mel_basis = np.mat(mel_basis)
    mel_basis_I = mel_basis.I
    mel_spec = []

    for i in range(len(mel)):
        print(mel_name)
        print(np.max(mel[i]))
        print(np.min(mel[i]))
        print(np.mean(mel[i]))
        #mel[i] = mel[i] * (0.6 / np.max(mel[i]))
        mel_db_item = np.transpose(mel[i])
        mel_db_item = denormalize_0_1(mel_db_item, hp.max_db, hp.min_db)
        #mel_db_item = np.maximum(mel_db_item, 0)
        # = normalize_0_1(mel_db_item, hp.default.max_db, hp.default.min_db)

        print(np.max(mel_db_item))
        print(np.mean(mel_db_item))

        mel_item = db2amp(mel_db_item)
        print(np.max(mel_item))

        mag_item = np.dot(mel_basis_I, mel_item)
        print(np.max(mel_item))
        mag_item = np.maximum(mag_item, 0)
        spec_item =  np.transpose(mag_item)

        #mag_db_item = amp2db(mag_item)
        #mag_db_item = normalize_0_1(mag_db_item, hp.default.max_db, hp.default.min_db)
        #mag_db_item = np.transpose(mag_db_item)
        #specitem = np.transpose(magitem)
        #mel_complex = mel_D_abs + np.complex(0, 0)
        #specitem = librosa.istft(stft_matrix=mel_complex, hop_length=hp.default.hop_length, win_length=hp.default.win_length)
        mel_spec.append(spec_item.getA())

    mel_spec = np.power(mel_spec, hp.emphasis_magnitude)
    mel_audio = np.array(list(map(lambda spec: spec2wav(spec.T, hp.n_fft, hp.win_length, hp.hop_length,
                                                 hp.n_iter), mel_spec)))

    mel_audio = inv_preemphasis(mel_audio, coeff=hp.preemphasis)
    tf.summary.audio(mel_name, mel_audio, hp.sr, max_outputs=hp.batch_size)
 

def main():
    g = S2SNet(is_training = False)
    g.loadg()

    test_path = "../data_thchs30/test-mini"
#    init_op = tf.global_variables_initializer()
#    g.sess.run(init_op)
    d = NetDataFlow(test_path)

    x_mels, x_specs, x_labels = d.get_data(4)
    sumimage(np.array(x_mels), 'x_mel')
    sumspecimage(np.array(x_specs), 'x_spec')

    decoded, ppgs, mel, spec = g.sess.run([g.decoded, g.ppgs, g.g_mel, g.g_spec], \
                                                   feed_dict={ g.x_mel: x_mels, g.x_spec : x_specs, g.x_label : x_labels })

    print(x_labels)
    print(decoded)

    f = open(test_path + "/phoneme.txt")
    words = []
    lines = f.readlines()
    for w in lines:
        words.append(w.replace('\n', ""))
        
    words_size = len(words)      
    word_num_map = dict(zip(words, range(len(words))))        # 当字符不在已经收集的words中时，赋予其应当的num，这是一个动态的结果      
    
    num_word_map = {}
    for k in word_num_map:
        num_word_map[word_num_map[k]] = k

    pingyin = ""
    
    for i in x_labels[0]:
        if i > 0 and i < hp.len_chinese_ppgs - 1:
            pingyin +=  num_word_map[i] + " "
    print(pingyin)

    pingyin = ""
    for i in decoded[0]:
        if i < hp.len_chinese_ppgs - 1:
            pingyin +=  num_word_map[i] + " "
    print(pingyin)

    pingyin = ""
    for i in range(len(ppgs[0])):
        pos = np.argmax(ppgs[0][i])
        if pos > 0 and pos < hp.len_chinese_ppgs -1  : # and ppgs[0][i][pos] > 0.1:
            pingyin +=  num_word_map[pos] + " "
        else:
            pingyin += "-"
    print(pingyin)
    


    sumimage(np.array(mel), 'g_mel')
    sumspecimage(np.array(spec), 'g_spec')

    summ = g.sess.run(tf.summary.merge_all(), feed_dict={ g.x_mel: x_mels, g.x_spec : x_specs, g.x_label : x_labels })
    g.writer.add_summary(summ)
    g.writer.close()
    print('done')
    
if __name__ == "__main__":
    main()


