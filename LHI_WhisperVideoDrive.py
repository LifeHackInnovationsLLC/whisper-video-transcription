# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %%
# LHI_WhisperVideoDrive.py
# %%
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/LifeHackInnovationsLLC/whisper-video-transcription/blob/main/LHI_WhisperVideoDrive.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="1Z6Z6Z6ePz1R"
# ### Jupytext Initialization (Sync Logic)
# Ensure Jupytext is installed and the notebook is paired with the `.py` file.
#
# import subprocess
# import sys
#
# def ensure_module(module_name, install_name=None):
#     """Install a module if it's not already installed."""
#     try:
#         __import__(module_name)
#         print(f"Module '{module_name}' is already installed.")
#     except ImportError:
#         install_name = install_name or module_name
#         print(f"Module '{module_name}' not found. Installing...")
#         subprocess.run([sys.executable, "-m", "pip", "install", install_name], check=True)
#
# Ensure Jupytext is installed
# ensure_module("jupytext")
#
# Sync the notebook with its paired `.py` file
# try:
#     subprocess.run(["jupytext", "--sync", "LHI_WhisperVideoDrive.ipynb"], check=True)
#     print("Jupytext synchronization successful.")
# except subprocess.CalledProcessError as e:
#     print(f"Error during Jupytext synchronization: {e}")


# %% Ensure required modules are installed and imported
# Handle missing modules and Google Colab environment checks

import subprocess
import sys

# Install and import required modules
required_modules = {
    "google.colab": "google-colab",
    "whisper": "openai-whisper",
    "librosa": "librosa",
    "soundfile": "soundfile"
}

for module, install_name in required_modules.items():
    try:
        __import__(module)
        print(f"Module '{module}' is already installed.")
    except ImportError:
        print(f"Module '{module}' not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", install_name], check=True)

# Conditional import for Google Colab
try:
    from google.colab import drive
    print("Google Colab environment detected.")
except ImportError:
    print("Google Colab environment not detected. Skipping Colab imports.")

# Import other required modules
import whisper
import librosa
import soundfile as sf



# %% [markdown] id="1Z6Z6Z6ePz1R"
#
# #📼 OpenAI Whisper + Google Drive Video Transcription
#
# 📺 Getting started video: https://youtu.be/YGpYinji7II
#
# ###This application will extract audio from all the video files in a Google Drive folder and create a high-quality transcription with OpenAI's Whisper automatic speech recognition system.
#
# *Note: This requires giving the application permission to connect to your drive. Only you will have access to the contents of your drive, but please read the warnings carefully.*
#
# This notebook application:
# 1. Connects to your Google Drive when you give it permission.
# 2. Creates a WhisperVideo folder and three subfolders (ProcessedVideo, AudioFiles and TextFiles.)
# 3. When you run the application it will search for all the video files (.mp4, .mov, mkv and .avi) in your WhisperVideo folder, transcribe them and then move the file to WhisperVideo/ProcessedVideo and save the transcripts to WhisperVideo/TextFiles. It will also add a copy of the new audio file to WhisperVideo/AudioFiles
#
# ###**For faster performance set your runtime to "GPU"**
# *Click on "Runtime" in the menu and click "Change runtime type". Select "GPU".*
#
#
# **Note: If you add a new file after running this application you'll need to remount the drive in step 1 to make them searchable**

# %% [markdown] id="DHhCHnaeTYw8"
# ##0. Choose which 'LHI Client' or folder to add transcriptions to

# %% id="L20Y96kiPz1R" colab={"base_uri": "https://localhost:8080/"} outputId="b7bdd274-635d-4d43-d73b-154b9d62d148"
import os

# Reusable function to check and mount Google Drive
def check_and_mount_drive():
    print("Checking /content/drive status...")
    if os.path.exists("/content/drive"):
        print("Mount directory exists. Checking contents...")
        if os.listdir("/content/drive"):
            print("Mountpoint already contains files. Attempting to unmount...")
            try:
                # Unmount the existing mountpoint
                # !fusermount -u /content/drive
                print("Unmounted successfully.")
            except Exception as e:
                print(f"Failed to unmount: {e}")
                print("If the issue persists, please select 'Runtime > Disconnect and delete runtime' and try again.")
                return False

    # Mount Google Drive
    print("Mounting Google Drive...")
    from google.colab import drive
    try:
        drive.mount("/content/drive", force_remount=True)
        print("Google Drive mounted successfully.")
    except ValueError as e:
        print(f"Mounting failed: {e}")
        print("If the issue persists, please select 'Runtime > Disconnect and delete runtime' and try again.")
        return False

    # Verify mount
    if os.path.exists("/content/drive/MyDrive"):
        print("Drive is mounted and ready.")
        return True
    else:
        print("Mounting seems incomplete. Please check your drive configuration.")
        return False

# Attempt to check and mount the drive
if check_and_mount_drive():
    print("Proceeding with folder creation...")
else:
    print("Drive mount failed. Cannot proceed.")
    raise SystemExit("Drive mount failed. Exiting.")

# Predefined options for client folders
clients = {
    "1": "/content/drive/MyDrive/Clients/WCBradley/Videos/",
    "2": "/content/drive/MyDrive/Clients/SiriusXM/Videos/",
    "3": "/content/drive/MyDrive/Clients/LHI/Videos/"
}

# Display options to the user
print("Select a client folder:")
print("1: WCBradley")
print("2: SiriusXM")
print("3: LHI")
print("4: Enter a custom folder path")

# Get user input
choice = input("Enter the number corresponding to your choice (default: 1): ").strip()

# Determine the root folder for the client
if choice in clients:
    client_videos_folder = clients[choice]
elif choice == "4":
    client_videos_folder = input("Enter the full path to your Videos folder: ").strip()
else:
    # Default to WCBradley if no valid input
    client_videos_folder = clients["1"]

# Define the WhisperVideo root folder within the client's Videos folder
rootFolder = client_videos_folder + "WhisperVideo/"

# Define subfolder paths relative to the WhisperVideo root folder
audio_folder = rootFolder + "AudioFiles/"
text_folder = rootFolder + "TextFiles/"
processed_folder = rootFolder + "ProcessedVideo/"

# Ensure WhisperVideo folder and its subfolders exist
folders = [rootFolder, audio_folder, text_folder, processed_folder]
for folder in folders:
    try:
        print(f"Checking folder: {folder}")
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
        else:
            print(f"Folder already exists: {folder}")
    except Exception as e:
        print(f"Error ensuring folder {folder}: {e}")

print(f"WhisperVideo folder and subfolders initialized for client:")
print(f"WhisperVideo folder: {rootFolder}")
print(f"Audio files folder: {audio_folder}")
print(f"Text files folder: {text_folder}")
print(f"Processed videos folder: {processed_folder}")


# %% [markdown] id="pFx0mfr031aw"
# ##1. Load the code libraries

# %% id="PomTPiCR5ihc" colab={"base_uri": "https://localhost:8080/"} outputId="bcb695ef-5838-4815-c862-132c09817166"
# !pip install git+https://github.com/openai/whisper.git
# !sudo apt update && sudo apt install ffmpeg
# !pip install librosa
# !pip install audioread

import whisper
import time
import librosa
import soundfile as sf
import re
import os

# model = whisper.load_model("tiny.en")
model = whisper.load_model("base.en")
# model = whisper.load_model("small.en") # load the small model
# model = whisper.load_model("medium.en")
# model = whisper.load_model("large")

# %% [markdown] id="JIjETRxb5nuE"
# ##2. Give the application permission to mount the drive and create the folders

# %% id="zxWvhDHzmspd"
# # Mount Google Drive
# from google.colab import drive
# drive.mount("/content/drive", force_remount=True)  # This will prompt for authorization.

# import os

# # Ensure WhisperVideo folder and its subfolders exist
# folders = [rootFolder, audio_folder, text_folder, processed_folder]
# for folder in folders:
#     try:
#         if not os.path.exists(folder):
#             os.makedirs(folder)
#             print(f"Created folder: {folder}")
#         else:
#             print(f"Folder already exists: {folder}")
#     except Exception as e:
#         print(f"Error ensuring folder {folder}: {e}")

# print(f"All folders verified and ready under: {rootFolder}")


# %% [markdown] id="7fr8tBQy5Tvo"
# ##3. Upload any video files you want transcribed in the "WhisperVideo" folder in your Google Drive.

# %% [markdown] id="nCef9V2i392e"
# ## 4. Extract audio from the video files and create a transcription
#
# This step processes video files in the `WhisperVideo` folder by extracting audio, transcribing it, and saving the transcription in the `TextFiles` folder. The original video file is moved to the `ProcessedVideo` folder upon successful transcription.
#
# ### Shareable Links
# The shareable link for the processed video is generated based on its Google Drive file path. This method avoids additional API calls and assumes that files are already shared within your team. The constructed link can be found at the beginning of the transcription file.
#
# Example of a shareable link format:
# ```
# https://drive.google.com/file/d/<file_id>/view
# ```
#
#

# %% id="D_rB5M99nmhw" colab={"base_uri": "https://localhost:8080/"} outputId="59e251c7-81a2-4c92-ac70-ad4e94b25ad4"
import os
import shutil
from datetime import timedelta
import subprocess
import logging
import csv
from datetime import datetime
# Removed: from urllib.parse import quote

# Helper function to format time
def format_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))

# Removed the generate_shareable_link function

# Setup logging
logging.basicConfig(
    filename="processing_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Ensure folders are created
folders = [rootFolder, audio_folder, text_folder, processed_folder]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Initialize logs
success_log = []
error_log = []
skipped_log = []

# Process all video files in WhisperVideo folder
video_files = [f for f in os.listdir(rootFolder) if os.path.isfile(os.path.join(rootFolder, f))]

for video_file in video_files:
    # Exclude the report file from processing
    if video_file == "processing_report.txt":
        continue

    # Skip non-video files
    if not video_file.endswith((".mp4", ".mov", ".avi", ".mkv")):
        skipped_log.append((video_file, "Invalid format"))
        print(f"Skipped {video_file}: Invalid format.")
        continue

    # Define paths
    video_path = os.path.join(rootFolder, video_file)
    audio_path = os.path.join(audio_folder, video_file[:-4] + ".wav")
    text_path = os.path.join(text_folder, video_file[:-4] + ".txt")
    processed_path = os.path.join(processed_folder, video_file)

    try:
        print(f"Extracting audio for {video_file} to {audio_path}")
        # Extract audio
        try:
            # Attempt to load the audio using librosa
            y, sr = librosa.load(video_path, sr=16000)  # Load audio with 16 kHz sampling rate
            sf.write(audio_path, y, sr)  # Save audio as a WAV file
            print(f"Audio extraction successful using librosa for {video_file}")
        except Exception as e_librosa:
            print(f"Librosa extraction failed for {video_file}: {e_librosa}")
            print(f"Falling back to ffmpeg for {video_file}")
            # Use ffmpeg as a fallback
            subprocess.run(["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", audio_path], check=True)
            print(f"Audio extraction successful using ffmpeg for {video_file}")

        print(f"Starting transcription for {audio_path}")
        # Transcribe the audio using Whisper
        result = model.transcribe(audio_path)
        print(f"Transcription completed for {audio_path}")

        # Create initial transcription
        text = ""
        for segment in result["segments"]:
            start_time = format_time(segment["start"])
            end_time = format_time(segment["end"])
            text_segment = segment["text"].strip()
            text += f"[{start_time} - {end_time}] {text_segment}\n\n"

        print(f"Saving transcription to {text_path}")
        # Save the transcription
        with open(text_path, "w") as f:
            f.write(text)
        print(f"Transcription saved successfully for {video_file}")

        print(f"Moving file {video_file} to processed folder")
        # Move the video to ProcessedVideo folder
        shutil.move(video_path, processed_path)
        print(f"File moved to processed folder: {processed_path}")

        # Log success
        success_log.append(video_file)
        logging.info(f"Successfully processed {video_file}")
        print(f"Successfully processed {video_file}")

    except subprocess.CalledProcessError as e:
        error_message = f"FFmpeg error for {video_file}: {e}"
        print(error_message)
        error_log.append((video_file, error_message))
        logging.error(error_message)
    except Exception as e:
        error_message = f"General error for {video_file}: {e}"
        print(error_message)
        error_log.append((video_file, error_message))
        logging.error(error_message)

# Perform folder parity check
def get_file_bases(folder):
    return {os.path.splitext(f)[0] for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))}

videos = get_file_bases(processed_folder)
audios = get_file_bases(audio_folder)
texts = get_file_bases(text_folder)

all_match = videos == audios == texts

# Generate completion report
report = "Processing Report\n"
report += f"\nSuccessfully Processed Files ({len(success_log)}):\n"
report += "\n".join(success_log)

report += f"\n\nSkipped Files ({len(skipped_log)}):\n"
report += "\n".join([f"{file} - {reason}" for file, reason in skipped_log])

report += f"\n\nErrors ({len(error_log)}):\n"
report += "\n".join([f"{file} - {reason}" for file, reason in error_log])

report += f"\n\nFolder Parity Check:\n"
report += f"All folders have matching files: {'Yes' if all_match else 'No'}\n"
report += f"Processed Videos: {len(videos)}\n"
report += f"Audio Files: {len(audios)}\n"
report += f"Text Files: {len(texts)}\n"

# Save the report
report_path = os.path.join(rootFolder, "processing_report.txt")
with open(report_path, "w") as f:
    f.write(report)

# Display completion report
print(report)

csv_path = os.path.join(rootFolder, "processing_log.csv")
file_exists = os.path.isfile(csv_path)

# We'll store a timestamp for each run and each file processed/skipped/error
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # If file doesn't exist, write header
    if not file_exists:
        writer.writerow(["Timestamp", "FileName", "Status", "Notes"])

    # Append rows for each processed file
    for fname in success_log:
        writer.writerow([current_time, fname, "Processed", ""])

    # Append rows for each skipped file
    for (fname, reason) in skipped_log:
        writer.writerow([current_time, fname, "Skipped", reason])

    # Append rows for each error file
    for (fname, reason) in error_log:
        writer.writerow([current_time, fname, "Error", reason])

print("\nCurrent CSV log entries:")
with open(csv_path, "r", encoding="utf-8") as csvfile:
    print(csvfile.read())


# %% id="OCvv85Y_u8V7"
# ### Jupytext Final Synchronization
# Ensure the notebook and `.py` file are in sync after all processing.

import subprocess
import sys

def sync_with_jupytext(notebook_file):
    """
    Ensure Jupytext is installed and synchronize the notebook with its paired `.py` file.
    """
    # Check if Jupytext is installed; install it if missing
    try:
        __import__('jupytext')
        print("Jupytext is already installed.")
    except ImportError:
        print("Jupytext not found. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "jupytext"], check=True)
            print("Jupytext installation successful.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing Jupytext: {e}")
            return

    # Attempt to synchronize the files
    try:
        print(f"Synchronizing notebook with '{notebook_file}'...")
        subprocess.run([sys.executable, "-m", "jupytext", "--sync", notebook_file], check=True)
        print("Synchronization successful: .ipynb and .py are now in sync.")
    except FileNotFoundError:
        print("Jupytext command not found. Skipping synchronization.")
    except subprocess.CalledProcessError as e:
        print(f"Error during Jupytext synchronization: {e}")

# Run synchronization
sync_with_jupytext("LHI_WhisperVideoDrive.ipynb")
