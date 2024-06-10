import argparse
import os
import pickle
import json
import librosa
import datetime
import time
import termcolor
import numpy as np

"""
Example of script call: 
`python3 tools/transform.py /mnt/e/Downloads/datasets/DATA/DATA/ /mnt/e/Downloads/datasets/DATA/NEW_GENERATED/ --noise_files /mnt/e/Downloads/datasets/DATA/NEW_GENERATED/ --desired_files /mnt/e/Downloads/datasets/DATA/NEW_GENERATED/ --depth 5`
`python3 tools/transform.py /mnt/e/Downloads/datasets/DATA/DATA /mnt/e/Downloads/datasets/DATA/NEW_GENARATED/ --simulate_masks`
"""

parser = argparse.ArgumentParser(description='NN GEV training - Combine audio files with masks/automate generating training files')

parser.add_argument('audio_files', help='Path to `.wav` audio files.')
parser.add_argument('output_path', help='Write results inside this path.')
parser.add_argument('--depth', help='Look this many folders deep inside provided folders for files (audio and masks both), default is: `5`', default=5)
parser.add_argument('--search_depth', help='Look for files inside folder structure, if `True`, provide desired depth, default is 5 folders!', action='store_true', default=False)
parser.add_argument('--simulate_masks', help='Only provide path to `audio_files`, will create random masks for both `noise` and `desired` signal.', action='store_true', default=False)

args = parser.parse_args()

# Parse script arguments:
simulate_masks: bool = args.simulate_masks
audio_path: str = args.audio_files
output_path: str = args.output_path

search_depth: bool = args.search_depth
depth: int = args.depth

def load_files(load_path: str, ext: str = "wav", depth: int = 0, tree: bool = False) -> list[str]:

    # Check if source folder exists:
    if not os.path.exists(load_path):
        print(f"Error, invalid input path: {termcolor.colored(load_path, 'green')}")
        exit(-1)

    # Final and step result lists:
    folder_content: list[str] = os.listdir(load_path)
    audio_files: list[str] = list()

    print(f"Searching for files, curr. depth: \t {termcolor.colored(depth, 'blue')}")

    # Check folder content:
    for file in folder_content:

        # If file ends with "ext" value, append it to result list:
        if not os.path.isdir(os.path.join(load_path, file)) and file.endswith(f".{ext}"):
            audio_files.append(os.path.join(load_path, file))

        # Search folder structure using recursion:
        elif tree:
            res = load_files(os.path.join(load_path, file), ext, depth + 1, tree = tree)
            audio_files.append(*res)

    return audio_files

# Generate random masks to test if training is possible / to test if environment is set up right:
def combine_with_random_masks(files: list[str], output_folder: str) -> bool:

    # Check if output folder exists, if not, create it:
    if not os.path.exists(output_folder):
        print(f"Folder: \"{termcolor.colored(output_folder, 'green')}\" does not exists, creating it now on same location...")
        os.mkdir(output_folder)
        print(f"Folder: \"{termcolor.colored(output_folder, 'green')}\" created!")

    # Load signal tensor, reshape it as algorithm require, and generate random IBM-s:
    for file in files:
        
        # Get original file name:
        original_signal_name: str = file.split(os.sep)[-1].split(".")[0]

        # Load file: 
        signal, _ = librosa.load(file, mono=False, dtype=np.float32)

        # Reshape signal:
        n_samples = signal.shape[1]
        
        segment_len = 513
        n_segments = n_samples // segment_len
        reshaped_audio = signal[:, :n_segments * segment_len].reshape(n_segments, signal.shape[0], segment_len)
        shape = reshaped_audio.shape

        # Generate 2 masks: sum of both masks is np.ones(signal.shape)
        noise_mask = np.random.rand(*shape).astype(np.float32)
        desired_mask = np.ones(shape=[*shape], dtype=np.float32) - noise_mask
        
        # Create dict with desired masks and original signal:
        file_content: dict = {
            "IBM_X": desired_mask,
            "IBM_N": noise_mask,
            "Y_abs": reshaped_audio
        }

        # Write to output folder:
        output_file = os.path.join(output_folder, f"{original_signal_name}.data")

        with open(output_file, "wb") as file_stream:
            pickle.dump(file_content, file_stream)

    return True


def combine_with_masks(audio_files: list[str], noise_masks: list[str], desired_masks: list[str], output_folder: str) -> bool:

    # Check if output folder exists, if not, create one:
    if not os.path.exists(output_folder):
        print(f"Folder: \"{termcolor.colored(output_folder, 'green')}\" does not exists, creating it now on same location...")
        os.mkdir(output_folder)
        print(f"Folder: \"{termcolor.colored(output_folder, 'green')}\" created!")

    # Main loop:
    for index, file in enumerate(audio_files):
        
        # Get unique name without extension:
        unique_name: str = file.split(".")[0]
        signal_name: str = file.split(os.sep)[-1].split(".")[0]

        noise_path: str = unique_name + ".noise"
        desired_path: str = unique_name + ".desired"

        # Load audio and mask data:
        audio, sr = librosa.load(file, mono=False, dtype=np.float32)
        noise, n_sr = librosa.load(noise_path, mono=False, dtype=np.float32)
        desired, d_sr = librosa.load(desired_path, mono=False, dtype=np.float32)

        # Reshape tensors:
        n_samples = audio.shape[1]
        segment_len = 513

        n_segments = n_samples // segment_len
        reshaped_audio = audio[:, :n_segments * segment_len].reshape(n_segments, audio.shape[0], segment_len)
        reshaped_noise = noise[:, :n_segments * segment_len].reshape(n_segments, noise.shape[0], segment_len)
        reshaped_desired = desired[:, :n_segments * segment_len].reshape(n_segments, desired.shape[0], segment_len)

        # Combine signals inside `dict` object:
        file_content: dict = {
            "IBM_X": reshaped_desired,
            "IBM_N": reshaped_noise,
            "Y_abs": reshaped_audio
        }

        # Write to output folder:
        output_file = os.path.join(output_folder, f"{signal_name}.data")

        with open(output_file, "wb") as file_stream:
            pickle.dump(file_content, file_stream)

    return True


# Generate JSON files used for training:
def generate_json_simple(source_folder: str, purpose: str = "tr", output_folder: str = None) -> str:

    # Check if source dir exists:
    if not os.path.exists(source_folder):
        print(f"Error, path: \"{termcolor.colored(source_folder, 'green')}\" does not exists!")
        return(-1)

    # Check if output dir exists, if not, create it:
    if not output_folder == None and os.path.exists(output_folder):
        print(f"Folder: \"{termcolor.colored(output_folder, 'green')}\" does not exists, creating it now on same location...")
        os.mkdir(output_folder)
        print(f"Folder: \"{termcolor.colored(output_folder, 'green')}\" created!")

    # Get all available files:
    files = os.listdir(source_folder)
    json_content = dict()

    index: int = 0
    # List all file paths:
    for file in files:
        if not os.path.isdir(os.path.join(source_folder, file)) and file.endswith(".data"):

            json_content[index] = os.path.join(source_folder, file)
            index += 1
    
    # Store json in output folder:
    output_path = source_folder if output_folder is None else output_folder
    output_path = os.path.join(output_path, f"flist_{purpose}.json")

    # Dump dict object to json file:
    with open(output_path, "w") as file_stream:
        json.dump(json_content, file_stream)

    return output_path

start = time.time()

# Load files, simulate IBM-s and generate `tr` and `dt` JSON files:
if simulate_masks:

    print(f"Generating files with random masks: \t {termcolor.colored(datetime.datetime.now().strftime('%H:%M:%S'), 'yellow')}")
    audio_files = load_files(audio_path, "wav", depth, search_depth)
    combine_with_random_masks(audio_files, output_path)

else:

    print(f"Combining audio files with existing masks: \t {termcolor.colored(datetime.datetime.now().strftime('%H:%M:%S'), 'yellow')}")
    audio_files = load_files(audio_path, "wav", depth, search_depth)
    noise_files = load_files(audio_path, "noise", depth, search_depth)
    desired_files = load_files(audio_path, "desired", depth, search_depth)

    combine_with_masks(audio_files, noise_files, desired_files, output_path)


# Generate JSON files used for training:
print(f"Generating JSON files: \t\t\t {termcolor.colored(datetime.datetime.now().strftime('%H:%M:%S'), 'yellow')}")
generate_json_simple(source_folder=output_path, purpose="tr")
generate_json_simple(source_folder=output_path, purpose="dt")
print(f"Done at: \t\t\t\t {termcolor.colored(datetime.datetime.now().strftime('%H:%M:%S'), 'magenta')}")

end = time.time()
print(f"Combining took: \t\t\t {termcolor.colored(str(end - start), 'magenta')}s")
