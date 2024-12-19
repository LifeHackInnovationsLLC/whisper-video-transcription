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

# %% [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/LifeHackInnovationsLLC/whisper-video-transcription/blob/main/LHI_WhisperVideoDrive.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% id="857e207c"
# LHI_WhisperVideoDrive.py
# %% id="f90aaf4f"
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

# %% [markdown] id="view-in-github"
# <a href="https://colab.research.google.com/github/LifeHackInnovationsLLC/whisper-video-transcription/blob/main/LHI_WhisperVideoDrive.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="5f7b404b"
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


# %% Ensure required modules are installed and imported colab={"base_uri": "https://localhost:8080/"} id="e425e731" outputId="88cd1d9d-3c68-4b7d-fef0-c21b5ce95d69"
# Handle missing modules and Google Colab environment checks

import subprocess
import sys


# Install and import required modules
required_modules = {
    "google.colab": "google-colab",
    "whisper": "openai-whisper",
    "librosa": "librosa",
    "soundfile": "soundfile",
    "colorama": "colorama",
    "google-api-python-client": "google-api-python-client",
    "google-auth-httplib2": "google-auth-httplib2",
    "google-auth-oauthlib": "google-auth-oauthlib"
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

# %% colab={"base_uri": "https://localhost:8080/"} id="L20Y96kiPz1R" outputId="107d8e38-a154-4a66-9be5-05f49633104f"
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

# %% colab={"base_uri": "https://localhost:8080/"} id="PomTPiCR5ihc" outputId="45a93879-07fa-4652-f974-abc3c4869ff2"
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

# %%
# Initialize colorama for console color support
init(autoreset=True)

# Google Drive API setup
def initialize_drive_api():
    """
    Initialize Google Drive API service account for generating shareable links.
    """
    try:
        credentials = Credentials.from_service_account_file(
            "/content/key.json",
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        service = build("drive", "v3", credentials=credentials)
        return service
    except Exception as e:
        print(Fore.RED + f"Failed to initialize Google Drive API: {e}")
        return None

drive_service = initialize_drive_api()


def get_file_id(file_name, folder_id):
    """
    Retrieve the file ID for a given file name in a specific folder on Google Drive.
    """
    try:
        results = drive_service.files().list(
            q=f"name='{file_name}' and '{folder_id}' in parents",
            spaces="drive",
            fields="files(id, name)",
            pageSize=1
        ).execute()
        items = results.get("files", [])
        if items:
            return items[0]["id"]
        else:
            print(Fore.YELLOW + f"File '{file_name}' not found in folder {folder_id}.")
            return None
    except Exception as e:
        print(Fore.RED + f"Error retrieving file ID for '{file_name}': {e}")
        return None


def generate_shareable_link(file_id):
    """
    Generate a shareable link for a given Google Drive file.
    """
    try:
        permission = {"type": "anyone", "role": "reader"}
        drive_service.permissions().create(fileId=file_id, body=permission).execute()
        link = f"https://drive.google.com/file/d/{file_id}/view"
        return link
    except Exception as e:
        print(Fore.RED + f"Failed to generate shareable link: {e}")
        return None



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

# %% colab={"base_uri": "https://localhost:8080/"} id="D_rB5M99nmhw" outputId="e6c4a64f-83e9-49e7-d865-640da49a8a35"
import os
import shutil
from datetime import timedelta
import subprocess
import logging
import csv
from datetime import datetime
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from colorama import Fore, Style, init
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
        print(Fore.YELLOW + f"Skipped {video_file}: Invalid format.")
        continue

    # Define paths
    video_path = os.path.join(rootFolder, video_file)
    audio_path = os.path.join(audio_folder, video_file[:-4] + ".wav")
    text_path = os.path.join(text_folder, video_file[:-4] + ".txt")
    processed_path = os.path.join(processed_folder, video_file)

    try:
        print(Fore.CYAN + f"Extracting audio for {video_file} to {audio_path}")
        # Extract audio
        try:
            # Attempt to load the audio using librosa
            y, sr = librosa.load(video_path, sr=16000)  # Load audio with 16 kHz sampling rate
            sf.write(audio_path, y, sr)  # Save audio as a WAV file
            print(Fore.GREEN + f"Audio extraction successful using librosa for {video_file}")
        except Exception as e_librosa:
            print(Fore.RED + f"Librosa extraction failed for {video_file}: {e_librosa}")
            print(Fore.YELLOW + f"Falling back to ffmpeg for {video_file}")
            # Use ffmpeg as a fallback
            subprocess.run(["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", audio_path], check=True)
            print(Fore.GREEN + f"Audio extraction successful using ffmpeg for {video_file}")

        print(Fore.CYAN + f"Starting transcription for {audio_path}")
        # Transcribe the audio using Whisper
        result = model.transcribe(audio_path)
        print(Fore.GREEN + f"Transcription completed for {audio_path}")

        # Create initial transcription
        text = ""
        for segment in result["segments"]:
            start_time = format_time(segment["start"])
            end_time = format_time(segment["end"])
            text_segment = segment["text"].strip()
            text += f"[{start_time} - {end_time}] {text_segment}\n\n"

        print(Fore.CYAN + f"Saving transcription to {text_path}")
        # Save the transcription
        with open(text_path, "w") as f:
            f.write(text)
        print(Fore.GREEN + f"Transcription saved successfully for {video_file}")

        print(Fore.CYAN + f"Moving file {video_file} to processed folder")
        # Move the video to ProcessedVideo folder
        shutil.move(video_path, processed_path)
        print(Fore.GREEN + f"File moved to processed folder: {processed_path}")

        # Retrieve file ID and generate shareable link
        file_id = get_file_id(video_file, processed_folder_id)
        if file_id:
            link = generate_shareable_link(file_id)
            if link:
                print(Fore.BLUE + f"Shareable Link: {link}")
            else:
                print(Fore.RED + f"Failed to generate shareable link for {video_file}")
        else:
            print(Fore.RED + f"Could not retrieve file ID for {video_file}")

        success_log.append(video_file)
        logging.info(f"Successfully processed {video_file}")
    except Exception as e:
        error_log.append((video_file, str(e)))
        logging.error(f"Error processing {video_file}: {e}")


        def get_or_create_folder(folder_name, parent_id=None):
            """
            Retrieve or create a folder in Google Drive.
            """
            try:
                # Search for the folder
                query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
                if parent_id:
                    query += f" and '{parent_id}' in parents"
                results = drive_service.files().list(
                    q=query,
                    spaces="drive",
                    fields="files(id, name)",
                    pageSize=1
                ).execute()
                items = results.get("files", [])
                if items:
                    return items[0]["id"]  # Return the existing folder ID
                else:
                    # Create the folder if it doesn't exist
                    folder_metadata = {
                        "name": folder_name,
                        "mimeType": "application/vnd.google-apps.folder",
                        "parents": [parent_id] if parent_id else []
                    }
                    folder = drive_service.files().create(body=folder_metadata, fields="id").execute()
                    return folder.get("id")
            except Exception as e:
                print(Fore.RED + f"Error creating or retrieving folder '{folder_name}': {e}")
                return None


        # Dynamically retrieve or create the ProcessedVideo folder
        processed_folder_id = get_or_create_folder("ProcessedVideo", parent_id=rootFolderID)
        if not processed_folder_id:
            print(Fore.RED + "Failed to retrieve or create ProcessedVideo folder. Exiting.")
            raise SystemExit("ProcessedVideo folder initialization failed.")





        # Retrieve file ID from Google Drive
        file_id = get_file_id(video_file, processed_folder_id)  # Assuming `processed_folder_id` is the ID of the ProcessedVideo folder
        if file_id:
            link = generate_shareable_link(file_id)
            if link:
                print(Fore.BLUE + f"Shareable Link: {link}")
            else:
                print(Fore.RED + f"Failed to generate shareable link for {video_file}")
        else:
            print(Fore.RED + f"Could not retrieve file ID for {video_file}")

        # Log success
        success_log.append(video_file)
        logging.info(f"Successfully processed {video_file}")
        print(Fore.GREEN + f"Successfully processed {video_file}")

    except subprocess.CalledProcessError as e:
        error_message = f"FFmpeg error for {video_file}: {e}"
        print(Fore.RED + error_message)
        error_log.append((video_file, error_message))
        logging.error(error_message)
    except Exception as e:
        error_message = f"General error for {video_file}: {e}"
        print(Fore.RED + error_message)
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


# %% colab={"base_uri": "https://localhost:8080/"} id="OCvv85Y_u8V7" outputId="e7b9b784-6cfd-42d4-e3a5-b0c2fdac1b05"
# ### Final Note for Synchronization
# For Colab: Sync changes manually after downloading the notebook.
# For Local: Use the Jupytext command:
#    jupytext --sync LHI_WhisperVideoDrive.ipynb

print("Final Note: Synchronize your files locally using Jupytext.")
print("Colab users: Save your notebook and download it to sync manually.")
