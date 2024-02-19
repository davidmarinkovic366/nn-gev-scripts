import os
import pickle 

data = None
with open('../../../mnt/e/Downloads/datasets/CHiME3_LDC2017S24.zip/CHiME3/data/audio/16kHz/isolated/dt05_bth/F01_22GC010A_BTH.CH0.wav', 'rb') as file:
    data = pickle.load(file)

print(data)