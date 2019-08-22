# -*- coding: utf-8 -*-
#/usr/bin/python2

import math

def get_T_y(duration, sr, hop_length, r):
    '''Calculates number of paddings for reduction'''
    def _roundup(x):
        return math.ceil(x * .1) * 10
    T = _roundup(duration*sr/hop_length)
    num_paddings = r - (T % r) if T % r != 0 else 0
    T += num_paddings
    return T

class Hyperparams:
    '''Hyper parameters'''
    # signal processing
    sr = 16000 # Sampling rate.
    n_fft = 1024 # fft points (samples)
    frame_shift = 0.0125 # seconds 0.0125
    frame_length = 0.05 # seconds 0.05
    hop_length = int(sr*frame_shift) # samples  This is dependent on the frame_shift.
    win_length = int(sr*frame_length) # samples This is dependent on the frame_length.
    n_mels = 80 # Number of Mel banks to generate
    n_mfcc = 40
    n_iter = 200 # Number of inversion iterations
    preemphasis = 0.97 # or None 0.97
    emphasis_magnitude = 1.4
    len_chinese_ppgs = 1210
    
    
    duration = 12
    max_db = 35
    min_db = -55

    hidden_units = 512  # alias: E
    slice_size = 20
    num_banks = 8
    num_highway_blocks = 8
    norm_type = 'ins'  # a normalizer function. value: bn, ln, ins, or None
    t = 1.0  # temperature
    dropout_rate = 0.2

    filters = 32

    # train
    batch_size = 8
    lr = 0.0003
    lr_cyclic_margin = 0.
    lr_cyclic_steps = 5000
    clip_value_max = 3.
    clip_value_min = -3.
    clip_norm = 10
    
    num_epochs = 10000
    steps_per_epoch = 10
    save_per_epoch = 50
    test_per_epoch = 1

    f_mode_dir = 'fmod/'
    g_mode_dir = 'gmod/'
 


