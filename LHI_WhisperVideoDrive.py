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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import shutil
import subprocess
import logging
import csv
from datetime import datetime, timedelta

import librosa
import soundfile as sf
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from colorama import Fore, Style, init
#Revision 1

# Initialize colorama for console color support
init(autoreset=True)

# -------------------------------------------------------------------------------------
# Prerequisites:
# - Variables: rootFolder, audio_folder, text_folder, processed_folder
# - drive_service: a valid Google Drive API service instance
# - model: a loaded Whisper model, e.g. model = whisper.load_model("base.en")
# - get_file_id(file_name, folder_id): function returning a file ID given name & parent folder ID
# - generate_shareable_link(file_id): function returning a shareable link given a file ID
#
# Make sure these are defined before running this code cell.
# -------------------------------------------------------------------------------------

# "myDriveID" refers to the top-level folder in Google Drive, i.e. "My Drive"
myDriveID = "root"

# Global dictionaries to keep track of entities
# entities_by_id[id] = {"path": path, "type": "file" or "folder", "url": url}
# entities_by_path[path] = {"id": id, "type": "file" or "folder", "url": url}
entities_by_id = {}
entities_by_path = {}

def get_gdrive_url(entity_type, entity_id):
    """Return a Google Drive URL for a file or folder given its type and ID."""
    if entity_type == "folder":
        return f"https://drive.google.com/drive/folders/{entity_id}"
    elif entity_type == "file":
        return f"https://drive.google.com/file/d/{entity_id}/view"
    else:
        return None

def register_entity(entity_id, path, entity_type):
    """Register a file or folder entity by its ID, path, and type."""
    if entity_type in ["file", "folder"]:
        url = get_gdrive_url(entity_type, entity_id)
    elif entity_type.startswith("local"):
        # Local file that doesn't exist on Drive
        url = f"local://{path}"
        entity_type = "file" if entity_type == "local_file" else entity_type
    else:
        # Unknown type fallback
        url = f"local://{path}"

    entities_by_id[entity_id] = {"path": path, "type": entity_type, "url": url}
    entities_by_path[path] = {"id": entity_id, "type": entity_type, "url": url}

def log_entity_action(prefix, entity_id=None, path=None):
    """
    Log an action involving an entity.
    If provided only entity_id or only path, attempt to resolve the other via the global maps.
    Print in the format:
      {Prefix} [Type] ID: {id}, Path: {path}, URL: {url}
    """
    final_id = entity_id
    final_path = path
    final_type = None
    final_url = None

    # If we have only an ID, try to find its path, type, and url
    if entity_id and entity_id in entities_by_id:
        final_path = entities_by_id[entity_id]["path"]
        final_type = entities_by_id[entity_id]["type"]
        final_url = entities_by_id[entity_id]["url"]

    # If we have only a path, try to find its ID and type
    if path and path in entities_by_path:
        final_id = entities_by_path[path]["id"]
        final_type = entities_by_path[path]["type"]
        final_url = entities_by_path[path]["url"]

    if not final_type:
        final_type = "unknown_type"
    if not final_id:
        final_id = "UNKNOWN_ID"
    if not final_path:
        final_path = "UNKNOWN_PATH"
    if not final_url:
        final_url = "unknown_url"

    print(Fore.CYAN + f"{prefix} [{final_type}] ID: {final_id}, Path: {final_path}, URL: {final_url}")

def format_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))

logging.basicConfig(
    filename="processing_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def get_or_create_folder(drive_service, folder_name, parent_id):
    """
    Retrieve or create a folder in Google Drive given a name and parent folder ID.
    """
    try:
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents"
        results = drive_service.files().list(
            q=query,
            spaces="drive",
            fields="files(id, name)",
            pageSize=1
        ).execute()
        items = results.get("files", [])

        if items:
            folder_id = items[0]["id"]
            # Use local path notation to store; can be pseudo
            # We'll refine paths if needed using get_folder_id_from_path later.
            pseudo_path = f"/GDriveVirtual/{folder_name}"
            register_entity(folder_id, pseudo_path, "folder")
            log_entity_action("Folder retrieved", entity_id=folder_id)
            return folder_id
        else:
            folder_metadata = {
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_id]
            }
            folder = drive_service.files().create(body=folder_metadata, fields="id").execute()
            folder_id = folder.get("id")
            pseudo_path = f"/GDriveVirtual/{folder_name}"
            register_entity(folder_id, pseudo_path, "folder")
            log_entity_action("Folder created", entity_id=folder_id)
            return folder_id
    except Exception as e:
        print(Fore.RED + f"Error creating or retrieving folder '{folder_name}': {e}")
        return None

def get_folder_id_from_path(drive_service, local_path):
    """
    Convert a local path into a Google Drive folder ID by traversing
    the folder tree starting from 'myDriveID'.
    """
    prefix = "/content/drive/MyDrive/"
    if not local_path.startswith(prefix):
        print(Fore.RED + "The path does not start with /content/drive/MyDrive/.")
        return None

    relative_path = local_path[len(prefix):].strip("/")
    if not relative_path:
        # It's just My Drive
        register_entity(myDriveID, "/content/drive/MyDrive/", "folder")
        log_entity_action("MyDrive is root", entity_id=myDriveID)
        return myDriveID

    path_parts = relative_path.split("/")
    current_parent_id = myDriveID
    current_path = "/content/drive/MyDrive/"
    for part in path_parts:
        current_path = os.path.join(current_path, part)
        folder_id = get_or_create_folder(drive_service, part, current_parent_id)
        if not folder_id:
            print(Fore.RED + f"Failed to navigate/create the folder for part: {part}")
            return None

        # Update the entity registered for this folder to reflect actual local_path
        if folder_id in entities_by_id:
            entities_by_id[folder_id]["path"] = current_path
        if current_path in entities_by_path:
            entities_by_path[current_path]["id"] = folder_id
            entities_by_path[current_path]["type"] = "folder"
            entities_by_path[current_path]["url"] = get_gdrive_url("folder", folder_id)
        else:
            register_entity(folder_id, current_path, "folder")

        log_entity_action("Folder confirmed", entity_id=folder_id)
        current_parent_id = folder_id

    return current_parent_id

# Ensure required local folders exist
folders = [rootFolder, audio_folder, text_folder, processed_folder]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(Fore.GREEN + f"Created folder: {folder}")
    else:
        print(Fore.GREEN + f"Folder exists: {folder}")

rootFolderID = get_folder_id_from_path(drive_service, rootFolder)
if not rootFolderID:
    print(Fore.RED + f"Failed to retrieve the Drive folder ID for {rootFolder}. Exiting.")
    raise SystemExit("rootFolder initialization failed.")
else:
    log_entity_action("rootFolderID retrieved", entity_id=rootFolderID)

processed_folder_id = get_or_create_folder(drive_service, "ProcessedVideo", parent_id=rootFolderID)
if not processed_folder_id:
    print(Fore.RED + "Failed to retrieve or create ProcessedVideo folder. Exiting.")
    raise SystemExit("ProcessedVideo folder initialization failed.")
else:
    log_entity_action("ProcessedVideo folder ID", entity_id=processed_folder_id)

success_log = []
error_log = []
skipped_log = []

video_files = [f for f in os.listdir(rootFolder) if os.path.isfile(os.path.join(rootFolder, f))]

for video_file in video_files:
    # Exclude the report file
    if video_file == "processing_report.txt":
        continue

    if not video_file.endswith((".mp4", ".mov", ".avi", ".mkv")):
        skipped_log.append((video_file, "Invalid format"))
        print(Fore.YELLOW + f"Skipped {video_file}: Invalid video format.")
        continue

    video_path = os.path.join(rootFolder, video_file)
    audio_path = os.path.join(audio_folder, video_file[:-4] + ".wav")
    text_path = os.path.join(text_folder, video_file[:-4] + ".txt")
    processed_path = os.path.join(processed_folder, video_file)

    # Register local file before we have a Drive ID
    register_entity("LOCAL_"+video_file, video_path, "local_file")

    try:
        print(Fore.CYAN + f"Extracting audio for {video_file} to {audio_path}")
        try:
            y, sr = librosa.load(video_path, sr=16000)
            sf.write(audio_path, y, sr)
            print(Fore.GREEN + f"Audio extraction successful using librosa for {video_file}")
        except Exception as e_librosa:
            print(Fore.RED + f"Librosa extraction failed for {video_file}: {e_librosa}")
            print(Fore.YELLOW + f"Falling back to ffmpeg for {video_file}")
            subprocess.run(["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", audio_path], check=True)
            print(Fore.GREEN + f"Audio extraction successful using ffmpeg for {video_file}")

        print(Fore.CYAN + f"Starting transcription for {audio_path}")
        result = model.transcribe(audio_path)
        print(Fore.GREEN + f"Transcription completed for {audio_path}")

        transcription_text = ""
        for segment in result["segments"]:
            start_time = format_time(segment["start"])
            end_time = format_time(segment["end"])
            text_segment = segment["text"].strip()
            transcription_text += f"[{start_time} - {end_time}] {text_segment}\n\n"

        print(Fore.CYAN + f"Saving transcription to {text_path}")
        with open(text_path, "w") as f:
            f.write(transcription_text)
        print(Fore.GREEN + f"Transcription saved successfully for {video_file}")

        print(Fore.CYAN + f"Moving file {video_file} to processed folder")
        shutil.move(video_path, processed_path)
        # Update the entity path since the file moved
        if "LOCAL_"+video_file in entities_by_id:
            entities_by_id["LOCAL_"+video_file]["path"] = processed_path
            # Update the URL for local file (still local until we get a file ID)
            entities_by_id["LOCAL_"+video_file]["url"] = f"local://{processed_path}"
        if video_path in entities_by_path:
            del entities_by_path[video_path]
        register_entity("LOCAL_"+video_file, processed_path, "local_file")
        log_entity_action("File moved to processed folder", path=processed_path)

        # Now try to get the file ID in the processed folder
        print(Fore.CYAN + f"Searching for file '{video_file}' in folder ID '{processed_folder_id}'...")
        file_id = get_file_id(video_file, processed_folder_id)
        if file_id:
            # Update the entity to reflect we have an actual Drive file ID now
            if "LOCAL_"+video_file in entities_by_id:
                # Remove the local reference
                del entities_by_id["LOCAL_"+video_file"]
            if processed_path in entities_by_path:
                del entities_by_path[processed_path]
            register_entity(file_id, processed_path, "file")
            log_entity_action("File found by ID", entity_id=file_id)

            link = generate_shareable_link(file_id)
            if link:
                print(Fore.BLUE + f"Shareable Link: {link}")
                with open(text_path, "a") as f:
                    f.write(f"\nOriginal Video Link: {link}\n")
            else:
                print(Fore.RED + f"Failed to generate shareable link for {video_file}")
        else:
            print(Fore.RED + f"Could not retrieve file ID for {video_file}")

        success_log.append(video_file)
        logging.info(f"Successfully processed {video_file}")

    except subprocess.CalledProcessError as ffmpeg_error:
        error_message = f"FFmpeg error for {video_file}: {ffmpeg_error}"
        print(Fore.RED + error_message)
        error_log.append((video_file, error_message))
        logging.error(error_message)

    except Exception as general_error:
        error_message = f"General error for {video_file}: {general_error}"
        print(Fore.RED + error_message)
        error_log.append((video_file, error_message))
        logging.error(error_message)

def get_file_bases(folder):
    return {os.path.splitext(f)[0] for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))}

videos = get_file_bases(processed_folder)
audios = get_file_bases(audio_folder)
texts = get_file_bases(text_folder)

all_match = (videos == audios == texts)

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

report_path = os.path.join(rootFolder, "processing_report.txt")
with open(report_path, "w") as f:
    f.write(report)

print(Fore.CYAN + "=== COMPLETION REPORT ===")
print(report)

csv_path = os.path.join(rootFolder, "processing_log.csv")
file_exists = os.path.isfile(csv_path)
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(["Timestamp", "FileName", "Status", "Notes"])

    for fname in success_log:
        writer.writerow([current_time, fname, "Processed", ""])
    for (fname, reason) in skipped_log:
        writer.writerow([current_time, fname, "Skipped", reason])
    for (fname, reason) in error_log:
        writer.writerow([current_time, fname, "Error", reason])

print(Fore.CYAN + "\nCurrent CSV log entries:")
with open(csv_path, "r", encoding="utf-8") as csvfile:
    print(csvfile.read())

print("Final Note: Synchronize your files locally using Jupytext.")
print("Colab users: Save your notebook and download it to sync manually.")
