import os
import json
import pickle
import numpy as np
import librosa
from matplotlib import pyplot as plt
import scipy.io.wavfile

train_path = '/mnt/e/Downloads/datasets/proba/train_output/flist_tr.json'

data = None
with open(train_path) as file_stream:
    data = json.load(file_stream)

# for key in data.keys():
#     print(f"{key}: {data[key]}")

test_instance = data["0"]
print(test_instance)

loaded = None
with open(test_instance, "rb") as file_stream:
    loaded = pickle.load(file_stream)

audio = loaded['Y_abs']
print(type(audio))
print(audio.shape)