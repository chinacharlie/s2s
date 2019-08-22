
import os
import sys
import librosa
import wave
import numpy as np


wav_path = '../data_thchs30/stf_mp3'  

def get_wav_files(wav_path=wav_path):      
    wav_files = []      
    for (dirpath, dirnames, filenames) in os.walk(wav_path):          
        for filename in filenames:              
            if filename.endswith(".wav") or filename.endswith(".WAV"):                  
                filename_path = os.sep.join([dirpath, filename])                  
                #if os.stat(filename_path).st_size < 100000:                      
                #    continue                  
                wav_files.append(filename_path)        
    return wav_files    

def trim_audio(wav_file):

    wav, sr = librosa.load(wav_file, sr=16000)
    partition_len = 1600

    partition = len(wav) // partition_len 

    head = 3
    for head in range(partition):
        tmp = np.sum(np.square(wav[head * partition_len : head * partition_len + partition_len]))
        #print("head %f" % tmp)
        if tmp < 0.0008:
            break

    if partition < 1:
        return 0

    end = partition -1 
    for end in range(partition -1, head, -1):
        a = np.sum(np.square(wav[end * partition_len : end * partition_len + partition_len]))
        #print("end = %f" % a)
        if a < 0.0008:
            break

    if end - head > (sr / partition_len * 5):
        wav = wav[head * partition_len : (end + 1) * partition_len]        
        newwav = np.zeros((sr * 10 - len(wav)), dtype=np.float32)
        newwav = np.concatenate((wav, newwav))

        f = wave.open(wav_file + ".trim.wav","wb")
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        newwav = newwav * (120 * 256 / np.max(newwav)) 
        newwav = newwav.astype(np.short)
        f.writeframes(newwav.tostring())
        f.close()
        print(wav_file + "trim.wav")
        return 1

    return 0


wav_files = get_wav_files()

total = len(wav_files)
process = 0
getcount = 0
for f in wav_files:
    process +=1
    getcount = getcount + trim_audio(f)
    print("%d/%d/%d %s" % (getcount, process, total, f))






    



