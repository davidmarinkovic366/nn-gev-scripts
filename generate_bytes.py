import os
import pickle
import numpy as np

# For test only, used to simulate 512 element ndarray that represent noise and desired IBM

encoded_x = np.random.random(512)
encoded_y = np.random.random(512)

print(type(encoded_x))
print(encoded_x.shape)

for i in range(1, 9):
    
    with open(f"/mnt/e/Downloads/datasets/proba/train/INPUT_{i}.IBM_X", 'wb') as file_stream: 
        pickle.dump(encoded_x, file_stream)

    with open(f"/mnt/e/Downloads/datasets/proba/train/INPUT_{i}.IBM_N", 'wb') as file_stream: 
        pickle.dump(encoded_y, file_stream)

    with open(f"/mnt/e/Downloads/datasets/proba/validation/INPUT_{i}.IBM_X", 'wb') as file_stream: 
        pickle.dump(encoded_x, file_stream)

    with open(f"/mnt/e/Downloads/datasets/proba/validation/INPUT_{i}.IBM_N", 'wb') as file_stream: 
        pickle.dump(encoded_y, file_stream)


print('done!')
    
