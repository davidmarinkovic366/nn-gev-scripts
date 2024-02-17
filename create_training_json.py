import os
import json
import pickle
import argparse
import librosa

# To combine all data in same folder, just set same path for 'train_output' and 'validation_output';

# Example of script call:
"""
python3 create_training_json.py /mnt/e/Downloads/datasets/proba/train /mnt/e/Downloads/datasets/proba/validation /mnt/e/Downloads/datasets/proba/train_output /mnt/e/Downloads/datasets/proba/validation_output
"""

parser = argparse.ArgumentParser(description='NN GEV training - JSON creation tool')

parser.add_argument('train_path', help='Path to training data')
parser.add_argument('validation_path', help='Path to validation data')
parser.add_argument('train_output', help='Path to output folder for generated training data and .json file')
parser.add_argument('validation_output', help='Path to output folder for generated validation data and .json file')

args = parser.parse_args()

def generate_json(name: str, unique_set: set[str], args) -> str:
    """
    Auto generate `.json` file based on provided training/validation data path;

    Args:
        - `name`: `'tr'` for training data, or `'dt'` for validation data
        - `unique_set`: set of unique file names, generated based on provided dataset folder
        - `args`: parser.parse_args()

    Returns:
        - `str` | `None`: Path to new `.json` file
    """

    # Generate .json file for training/validation data paths:
    json_content: dict = dict()

    # Training or validation data path:
    containing_path: str = args.train_path if name == 'tr' else args.validation_path

    # Loop over training/validation files and create json file:
    for i, file in enumerate(sorted(unique_set)):

        # Overall .json file, used for storing paths to train/validation data files
        json_content[i] = os.path.join(args.train_output if name == 'tr' else args.validation_output, file)

        # Data dict, used to combine input .wav file, Noise IBM and Desired IBM:        
        instance_dict: dict = dict()
        
        # Error with picke and .wav files, must be loaded this way:
        audio_data, _ = librosa.load(os.path.join(containing_path, file + ".wav"), mono=False)
        
        # Save loaded .wav file with pickle:
        instance_dict['Y_abs'] = audio_data

        # Loop over IBM-s for same input signal:
        for extension in ['IBM_N', 'IBM_X']:

            # All content is loaded as binary array:
            with open(os.path.join(containing_path, file + '.' + extension), 'rb') as file_stream:
                instance_dict[extension] = pickle.load(file_stream)

        # Store generated dict for INPUT_X (INPUT_X.wav, INPUT_X.IBM_N, INPUT_X.IBM_X)
        with open(os.path.join(args.train_output if name == 'tr' else args.validation_output, file), 'wb') as file_stream:
            pickle.dump(instance_dict, file_stream)

    # Generate output file path:
    output = args.train_output if name == 'tr' else args.validation_output
    output_path = os.path.join(output, f"flist_{name}.json")

    # Write dictionary to .json file:
    with open(output_path, 'w') as file_stream:
        json.dump(json_content, file_stream)

    # Return path to new .json file:
    return output_path


"""
File structure of train_path || validation_path:
    
    INPUT_1.wav         # input signal / before beamformer
    INPUT_1.IBM_N       # ideal noise binary mask
    INPUT_1.IBM_X       # ideal desired binary mask

    INPUT_2.wav
    INPUT_2.IBM_N
    INPUT_2.IBM_X
    
    ...
"""

# FIXME: for testing, remove in final version;
# print("\nInput args: ")
# print(f"Train path: {args.train_path}")
# print(f"Validation path: {args.validation_path}")
# print(f"Train output path: {args.train_output}")
# print(f"Validation output path: {args.validation_output}")
# print("=====================================================\n")

# Create output dirs if they don't exists:
if not os.path.exists(args.train_output):
    os.mkdir(args.train_output)

if not os.path.exists(args.validation_output):
    os.mkdir(args.validation_output)

# List for file names inside provided dataset dirs:
train_files = os.listdir(args.train_path)
validation_files = os.listdir(args.validation_path)

# Generate unique names list for each data object: ['INPUT_1', 'INPUT_2', 'INPUT_3', ..., 'INPUT_N']
train_unique: set[str] = set()
validation_unique: set[str] = set()

for file in train_files:
    train_unique.add(file.split('.')[0])

for file in validation_files:
    validation_unique.add(file.split('.')[0])

# FIXME: for testing, remove in final version; 
# print("Generated unique names: ")
# print(f"Unique train: \n{train_unique}")
# print(f"Unique validation: \n{validation_unique}")
# print("=====================================================\n")

print(f"\nSucessfully generated .json TRAINING file at: \n{generate_json('tr', train_unique, args)}")
print(f"Sucessfully generated .json VALIDATION file at: \n{generate_json('dt', validation_unique, args)}")
print("========================================================================\n")

print("Generating and combining audio files with binary masks is done, now you can run train.py script!")
