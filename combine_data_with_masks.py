import os
import pickle
import json
import librosa
import numpy as np

def load_files(load_path: str, depth: int = 0, tree: bool = False) -> list[str]:

    if not os.path.exists(load_path):
        print(f"Error, invalid input path: {load_path}")
        exit(-1)

    folder_content: list[str] = os.listdir(load_path)
    audio_files: list[str] = list()

    print(f"Searching for files, curr. depth: {depth}")

    for file in folder_content:

        if not os.path.isdir(os.path.join(load_path, file)):
            audio_files.append(os.path.join(load_path, file))

        elif tree:
            res = load_files(os.path.join(load_path, file), depth + 1, tree = tree)
            for r in res:
                audio_files.append(r)

    return audio_files

def combine_and_save(files: list[str], output_folder: str) -> bool:

    # Check if output folder exists, if not, create it;
    if not os.path.exists(output_folder):
        print(f"Folder: \"{output_folder}\" does not exists, creating it now on same location...")
        os.mkdir(output_folder)
        print(f"Folder \"{output_folder}\" created!")

    # Generate 2 masks for every signal in dataset:
    for file in files:
        
        # Get original file name:
        original_signal_name: str = file.split(os.sep)[-1].split(".")[0]

        # Load file: 
        signal, _ = librosa.load(file, mono=False, dtype=np.float32)

        n_samples = signal.shape[1]

        # Transform from [N, M] to [M // 513, N, 513]
        # M // 513 chunks of N channels, with 513 samples, last chunk from original signal is removed if shorter than 513 elements
        segment_len = 513
        n_segments = n_samples // segment_len
        reshaped_audio = signal[:, :n_segments * segment_len].reshape(n_segments, signal.shape[0], segment_len)
        shape = reshaped_audio.shape

        # TODO: remove in future!
        print(f"\"{file}\": \"{original_signal_name}\": {signal.shape}")

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

        print(f"File: \"{original_signal_name}\" is combined with masks!")

    return True

def generate_json_simple(source_folder: str, purpose: str = "tr", output_folder: str = None) -> str:

    if not os.path.exists(source_folder):
        print(f"Error, path: \"{source_folder}\" does not exists!")
        return(-1)

    if not output_folder == None and os.path.exists(output_folder):
        print(f"Output folder, does not exists, generating new one!")
        os.mkdir(output_folder)
        print(f"Created new folder: \"{output_folder}\"")

    # Get all available files:
    files = os.listdir(source_folder)
    json_content = dict()

    index: int = 0
    # List all file paths:
    for file in files:
        if not os.path.isdir(os.path.join(source_folder, file)) and file.endswith(".data"):

            # TODO: remove in future!
            # print(f"{index}:\t {os.path.join(source_folder, file)}")
            json_content[index] = os.path.join(source_folder, file)
            index += 1
    
    # Store json in output folder:
    output_path = source_folder if output_folder is None else output_folder
    output_path = os.path.join(output_path, f"flist_{purpose}.json")

    # Dump dict object to json file:
    with open(output_path, "w") as file_stream:
        json.dump(json_content, file_stream)

    return output_path

result = load_files("/mnt/e/Downloads/datasets/DATA/DATA/", 0, True)
is_done: bool = combine_and_save(files=result, output_folder="/mnt/e/Downloads/datasets/DATA/GENERATED/")
generated_json_path = generate_json_simple(source_folder="/mnt/e/Downloads/datasets/DATA/GENERATED/", purpose="tr")
generated_json_path = generate_json_simple(source_folder="/mnt/e/Downloads/datasets/DATA/GENERATED/", purpose="dt")

print("Done!" if is_done else "Error :(")
# for i, el in enumerate(result):
    # print(f"{i}:\t{el}")
