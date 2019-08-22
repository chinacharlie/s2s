# -*- coding: utf-8 -*-
# /usr/bin/python2

import glob
import random

import os
import sys
import librosa
import numpy as np
from audio import read_wav, preemphasis, amp2db
from hyperparams import Hyperparams as hp
from utils import normalize_0_1
from collections import Counter  
from joblib import Parallel, delayed    
import codecs 
import time

def wav_random_crop(wav, sr, duration):
    assert (wav.ndim <= 2)

    target_len = sr * duration
    wav_len = wav.shape[-1]
    start = np.random.choice(range(np.maximum(1, wav_len - target_len)), 1)[0]
    end = start + target_len
    if wav.ndim == 1:
        wav = wav[start:end]
    else:
        wav = wav[:, start:end]
    return wav


def get_mfccs_and_phones(wav_file, trim=False, random_crop=True):

    '''This is applied in `train1` or `test1` phase.
    '''

    # Load
    wav = read_wav(wav_file, sr=16000)#hp.sr)
    

    mfccs, _, _ = _get_mfcc_and_spec(wav, 0.97, 512, #hp.preemphasis, hp.n_fft,
                                     400, #hp.win_length,
                                     80)  #hp.hop_length)

    # timesteps
    num_timesteps = mfccs.shape[0]

    # phones (targets)
    phn_file = wav_file.replace("WAV", "PHN").replace("wav", "PHN")
    phn2idx, idx2phn = load_vocab()
    phns = np.zeros(shape=(num_timesteps,))
    bnd_list = []
    for line in open(phn_file, 'r').read().splitlines():
        start_point, _, phn = line.split()
        bnd = int(start_point) // 80 #hp.hop_length
        phns[bnd:] = phn2idx[phn]
        bnd_list.append(bnd)

    # Trim
    if trim:
        start, end = bnd_list[1], bnd_list[-1]
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Random crop
    n_timesteps = (3 * 16000) // 80 + 1# (hp.duration * hp.sr) // hp.hop_length + 1
    if random_crop:
        start = np.random.choice(range(np.maximum(1, len(mfccs) - n_timesteps)), 1)[0]
        end = start + n_timesteps
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Padding or crop
    mfccs = librosa.util.fix_length(mfccs, n_timesteps, axis=0)
    phns = librosa.util.fix_length(phns, n_timesteps, axis=0)

    return mfccs, phns


def get_mfccs_and_spectrogram(wav_file, trim=True, random_crop=False):
    '''This is applied in `train2`, `test2` or `convert` phase.
    '''


    # Load
    wav, _ = librosa.load(wav_file, sr=hp.sr)

    # Trim
    if trim:
        wav, _ = librosa.effects.trim(wav, frame_length=hp.win_length, hop_length=hp.hop_length)

    if random_crop:
        wav = wav_random_crop(wav, hp.sr, hp.duration)

    # Padding or crop
    length = hp.sr * hp.duration
    wav = librosa.util.fix_length(wav, length)

    return _get_mfcc_and_spec(wav, hp.preemphasis, hp.n_fft, hp.win_length, hp.hop_length)


# TODO refactoring
def _get_mfcc_and_spec(wav, preemphasis_coeff, n_fft, win_length, hop_length):

    # Pre-emphasis
    y_preem = preemphasis(wav, coeff=preemphasis_coeff)

    # Get spectrogram
    D = librosa.stft(y=y_preem, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(D)

    # Get mel-spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t) # mel spectrogram
    
    

    # Get mfccs, amp to db
    mag_db = amp2db(mag)
    mel_db = amp2db(mel)
    #mfccs = np.dot(librosa.filters.dct(hp.n_mfcc, mel_db.shape[0]), np.clip(mel_db, hp.max_db, hp.min_db) )
    #mfccs = np.maximum(mfccs, 0)
    mfccs = np.dot(librosa.filters.dct(hp.n_mfcc, mel_db.shape[0]), mel_db)
    # Normalization (0 ~ 1)
    mag_db = normalize_0_1(mag_db, hp.max_db, hp.min_db)
    mel_db = normalize_0_1(mel_db, hp.max_db, hp.min_db)
    


    return mfccs.T, mag_db.T, mel_db.T  # (t, n_mfccs), (t, 1+n_fft/2), (t, n_mels)



def get_wav_files(wav_path):      
    wav_files = []    
    print(os.getcwd() + wav_path) 
    for (dirpath, _, filenames) in os.walk( wav_path):          
        for filename in filenames:              
            if filename.endswith(".wav") or filename.endswith(".WAV"):                  
                filename_path = os.sep.join([dirpath, filename])                  
                if os.stat(filename_path).st_size < 200000:                      
                    continue                  
                wav_files.append(filename_path)        
    return wav_files    

def get_phoneme(trn_file):
        root_path = os.path.dirname(trn_file)
        if not os.path.isfile(trn_file):
            return ""

        f = codecs.open(trn_file, mode='r', encoding='utf-8')
        lines = f.readlines()
        if len(lines) > 1 :
            return lines[1].replace("\n", "")
        trn_file = root_path + "/" + lines[0].replace("\n", "")
        f = codecs.open(trn_file, mode='r', encoding='utf-8')
        trn = f.readlines()[1].replace("\n", "")
        return trn


def get_wav_label(wav_files):      
    labels_dict = {}    

    for f in wav_files:
        label_id = os.path.basename(f).split(".")[0]  
        label_text = get_phoneme(f + ".trn")
        labels_dict[label_id] = label_text.replace("\n", "")

    labels = []      
    new_wav_files = []      
    for f in wav_files:
        label_id = os.path.basename(f).split(".")[0]  
        if label_id in labels_dict:              
            labels.append(labels_dict[label_id])             
            new_wav_files.append(f) 

    return new_wav_files, labels    

class NetDataFlow:
    def __init__(self, data_path, is_cache = False, mel_clip = 0.00001):
        self.data_path = data_path
        self.mel_clip = mel_clip 
        
        self.wav_files = get_wav_files(self.data_path)
        self.wav_files, labels = get_wav_label(self.wav_files)      

        f = open(data_path + "/phoneme.txt")
        words = []
        lines = f.readlines()
        for w in lines:
            words.append(w.replace('\n', ""))
        
        words_size = len(words)      
        print(u"词汇表大小：", words_size)        
        self.word_num_map = dict(zip(words, range(len(words))))        # 当字符不在已经收集的words中时，赋予其应当的num，这是一个动态的结果      

        to_num = lambda word: self.word_num_map.get(word, len(words))        # 将单个file的标签映射为num 返回对应list,最终all file组成嵌套list      
        self.labels_vector = [list(map(to_num, label.split(" ") ) ) for label in labels]        
        self.label_max_len = np.max([len(label) for label in self.labels_vector])      
        print(u"最长句子的字数:" + str(self.label_max_len))   

        self.num_word_map = {}
        for k in self.word_num_map:
            self.num_word_map[self.word_num_map[k]] = k

        self.pointer = 10
        self.cache = {}
        self.is_cache = is_cache
    
    def get_phoneme(self, index):
        return self.num_word_map[index]
    
    def get_data(self, batch_size = hp.batch_size):

        mels = []
        specs = []
        labels = []
     
        for i in range(batch_size):

            self.pointer = random.randint(0, len(self.wav_files)-1)



            if self.pointer in self.cache:
                spec, mel = self.cache[self.pointer]

            else:    
                _, spec, mel = get_mfccs_and_spectrogram(self.wav_files[self.pointer]) 
                if self.is_cache:
                    self.cache[self.pointer] = [spec, mel]
            label = self.labels_vector[self.pointer]

            self.pointer += 5
                        
            while len(label) < self.label_max_len:              
                    label.append(hp.len_chinese_ppgs - 1)        
            mels.append(mel[:-1])
            specs.append(spec[:-1])
            labels.append(label)

        return mels, specs, labels
 

if __name__ == "__main__":
   
    df  = NetDataFlow('../data_thchs30/test-mini', True)
    for i in range(10):
        a, b, c = df.get_data()
        print(time.time())

    df = NetDataFlow('../data_thchs30/test-mini')
    for i in range(10):
        a, b, c = df.get_data()
        print(time.time())
    
#    print(a)
#    for i in a[0]:
#        print(i)
    




