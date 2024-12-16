
# Whisper Video Transcription

A project for transcribing videos using Whisper, integrated with Google Drive for file management and Google Colab for execution. The project maintains synchronization between Jupyter Notebooks and Python scripts using Jupytext.

## Table of Contents

- [Setup](#setup)
- [Usage](#usage)
  - [Local Development with PyCharm](#local-development-with-pycharm)
  - [Running in Google Colab](#running-in-google-colab)
- [Synchronization with Jupytext](#synchronization-with-jupytext)
- [Version Control](#version-control)
- [Contributing](#contributing)
- [License](#license)

## Setup

### Prerequisites

- Python 3.7 or higher
- Git
- [PyCharm Professional](https://www.jetbrains.com/pycharm/) (for enhanced Jupyter support)
- [Jupyter Notebook](https://jupyter.org/install) or [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
- Google Account for accessing Google Colab and Google Drive

### Clone the Repository

```bash
git clone https://github.com/LifeHackInnovationsLLC/whisper-video-transcription.git 
cd whisper-video-transcription
```

### Install Dependencies

It's recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** If `jupytext` is not listed in your `requirements.txt`, add it:

```bash
pip install jupytext
```

Then, update `requirements.txt`:

```bash
pip freeze > requirements.txt
```

## Usage

### Local Development with PyCharm

1. **Open the Project:**
   
   Open `whisper-video-transcription` in PyCharm.

2. **Ensure Jupytext is Installed:**
   
   If not already installed, install Jupytext in your virtual environment:

```bash
pip install jupytext
```

3. **Edit and Sync:**
   
   - **Editing:** Make changes in either `LHI_WhisperVideoDrive.ipynb` or `LHI_WhisperVideoDrive.py`.
   - **Syncing:** Use the terminal in PyCharm to run:

```bash
jupytext --sync LHI_WhisperVideoDrive.ipynb
```

4. **Version Control:**
   
   Commit both `.ipynb` and `.py` files to keep them in sync.

### Running in Google Colab

1. **Open the Notebook:**
   
   - Navigate to your GitHub repository in Colab by selecting **File > Open Notebook > GitHub** and entering your repository URL.
   - Alternatively, directly open the notebook in Colab via URL: `https://colab.research.google.com/github/your-username/whisper-video-transcription/blob/main/LHI_WhisperVideoDrive.ipynb`

2. **Ensure Jupytext is Installed:**
   
   The synchronization cells added to the notebook will handle this. When you run the notebook, the following cells will:

   - Install Jupytext.
   - Pair the notebook with the `.py` file.
   - Synchronize changes.

3. **Run the Notebook:**
   
   Execute all cells. This will ensure that the `.py` file is updated with any changes made in Colab.

4. **Synchronization Steps in Colab:**
   
   - **At the Start:** The initialization cells install Jupytext and sync the notebook.
   - **At the End:** The final cell syncs any changes back to the `.py` file.

## Synchronization with Jupytext

This project uses Jupytext to maintain synchronization between Jupyter Notebooks (`.ipynb`) and Python scripts (`.py`). This allows for:

- **Version Control:** Track changes more effectively using Git with `.py` files.
- **IDE Integration:** Use PyCharm’s powerful tools for code editing and comparison.
- **Collaborative Development:** Easily share and update code and documentation.

### Setting Up Jupytext

1. **Pair the Notebook with the `.py` File:**

   This has already been set up, but for future reference:

```bash
jupytext --set-formats ipynb,py:percent LHI_WhisperVideoDrive.ipynb
```

2. **Syncing Changes:**

   - **Locally (PyCharm):**

```bash
jupytext --sync LHI_WhisperVideoDrive.ipynb
```

   - **In Colab:**

     The synchronization cells in the notebook handle this automatically.

## Version Control

Both `LHI_WhisperVideoDrive.ipynb` and `LHI_WhisperVideoDrive.py` are tracked in Git. Ensure you commit changes to both files to maintain synchronization.

```bash
git add LHI_WhisperVideoDrive.ipynb LHI_WhisperVideoDrive.py
git commit -m "Describe your changes here"
git push origin main
```

## Contributing

Contributions are welcome! Please ensure that any changes maintain synchronization between the notebook and Python script by using Jupytext.

## License

[MIT License](LICENSE)

---

### **Important Steps to Remember**

1. **Synchronization Commands in Colab:**
   
   - **Initialization at the Start:**
     
     Ensure that every time you open the notebook in Colab, the synchronization cells are run to pair and sync the notebook with the `.py` file.

   - **Final Sync at the End:**
     
     Always run the final synchronization cell before ending your Colab session to ensure all changes are saved to the `.py` file.

2. **Commit Both Files to Git:**
   
   Always add and commit both `LHI_WhisperVideoDrive.ipynb` and `LHI_WhisperVideoDrive.py` to your Git repository after making changes. This ensures that the pairing metadata and the latest code are both version-controlled.

```bash
git add LHI_WhisperVideoDrive.ipynb LHI_WhisperVideoDrive.py
git commit -m "Update transcription logic and documentation"
git push origin main
```

3. **Running `jupytext --sync` Locally:**
   
   Whenever you make changes in PyCharm, run `jupytext --sync LHI_WhisperVideoDrive.ipynb` to ensure the `.py` file is up-to-date.

4. **Avoid Manual Edits in the `.py` File:**
   
   Make all changes through the notebook or the `.py` file, but always use Jupytext to keep them in sync. Manual edits without synchronization can lead to discrepancies.

5. **Handling Merge Conflicts:**
   
   When collaborating, if merge conflicts arise in either file, resolve them carefully by ensuring both files are consistent after the merge.

### **Final Workflow Overview**

1. **Editing in PyCharm:**
   
   - Open `LHI_WhisperVideoDrive.py` or `LHI_WhisperVideoDrive.ipynb`.
   - Make changes as needed.
   - Run `jupytext --sync LHI_WhisperVideoDrive.ipynb` in the PyCharm terminal to synchronize.
   - Commit both files to Git.

2. **Editing in Google Colab:**
   
   - Open the notebook from GitHub.
   - Run all cells, ensuring synchronization commands execute.
   - Make changes and run the notebook.
   - Ensure the final sync cell runs to update the `.py` file.
   - Commit both files to Git via your local machine or GitHub interface.

3. **Using ChatGPT for Assistance:**
   
   - Refer to the `.py` file for code snippets and documentation.
   - After receiving suggestions, apply them to the `.py` file.
   - Sync using Jupytext to reflect changes in the notebook.

By following these steps and maintaining clear synchronization practices, you can efficiently develop your project across both PyCharm and Google Colab without conflicts.

If you encounter any issues or need further assistance, feel free to ask!
