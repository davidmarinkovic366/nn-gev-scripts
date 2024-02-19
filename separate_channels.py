import argparse
import os
import librosa
import soundfile
import time
import threading

"""
Example of script call:

python3 separate_channels.py /mnt/e/Downloads/datasets/proba/train /mnt/e/Downloads/datasets/proba/train_output_separate --display_tree
"""

parser = argparse.ArgumentParser(description='Audio channel separation tool')

parser.add_argument('input_path', help='Path to training data')
parser.add_argument('output_path', help='Path to validation data')
parser.add_argument('--display_tree', help='Print output folder content in real time?', action='store_true', default=False)

args = parser.parse_args()

# Global variables:
global_counter: int = 0
max_depth: int = 1
display_tree = args.display_tree
animation_running: bool = False

def animate():
  """Animates dots in a separate thread."""
  global animation_running
  animation_chars = ".oO*."
  while animation_running:
    for char in animation_chars:
      print("\rProcessing..." + char, end="")
      time.sleep(0.2)
      print("\r", end="")  # Back to beginning of line

def generate_tree_output_dir(source_path: str, output_path: str, current_depth: int = 1) -> bool:
    """
    Copy file structure from provided source, inside output path, recursively

    Args:
        - `source_path`: Path to source containing folder 
        - `output_path`: Path to destination folder

    Returns:
        - `bool`: Successfully finished operation?
    """

    global global_counter
    global max_depth
    global display_tree

    # Calculate recursion depth:
    if max_depth < current_depth:
        max_depth = current_depth

    # Prepare output folder if necessary:
    if not os.path.exists(output_path):
        # print("Creating output folder: {output_path}")
        os.mkdir(output_path)

    # List content of provided folder:
    content = os.listdir(source_path)
    root_tab: str = (current_depth - 1) * "\t"

    if display_tree:
        print(f"{root_tab}{output_path.split(os.sep)[-1]}:")

    # DFS
    for file in content:

        # Will consider only .wav files, modify if necessary:
        if os.path.isfile(os.path.join(source_path, file)) and file.endswith(".wav"):
            
            # Separate audio in multiple files:
            loaded_audio, sample_rate = librosa.load(os.path.join(source_path, file), mono=False)
            file_name = file.split(".")[0]

            """
            input:  AUDIO.wav (4 ch)
            output:
                    AUDIO_CH_0.wav
                    AUDIO_CH_1.wav
                    AUDIO_CH_2.wav
                    AUDIO_CH_3.wav
            """

            # FIXME: Remove in future, just for testing purposes:
            # print(f"Processing file: {os.path.join(source_path, file)} \t shape: {loaded_audio.shape}")

            for i, channel in enumerate(loaded_audio):
                soundfile.write(os.path.join(output_path, file_name + f"_CH_{i}.wav"), channel, sample_rate)
                
                if display_tree:                
                    file_tab: str = current_depth * "\t"
                    print(file_tab + "- " + file_name + "_CH_" + f"{i}" + ".wav")

            global_counter += 1

        elif os.path.isdir(os.path.join(source_path, file)):
            
            # Generate path for new folder, and process it with same function;
            next_source_path: str = os.path.join(source_path, file)
            next_output_path: str = os.path.join(output_path, file)

            generate_tree_output_dir(source_path=next_source_path, output_path=next_output_path, current_depth=current_depth + 1)

    return True


def list_files(start_path):
    for root, dirs, files in os.walk(start_path):
        level = root.replace(start_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(sub_indent, f))

# Create output dir if necessary:
if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

# Just for fun :)
animation_thread = threading.Thread()
if not display_tree:
    animation_thread = threading.Thread(target=animate)
    animation_running = True
    animation_thread.start()

# Get all files and create tree folder structure if necessary:
generate_tree_output_dir(source_path=args.input_path, output_path=args.output_path, current_depth=1)

if not display_tree:
    animation_running = False
    animation_thread.join()

print(f"\nFinished separating channels in separate audio files!")
print(f"\t* Processed .wav files count:\t {global_counter}")
print(f"\t* Max folder depth:\t\t {max_depth}")
