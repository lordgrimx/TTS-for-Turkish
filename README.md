# Text-to-Speech Model Training

This repository contains code for training a Text-to-Speech (TTS) model using the Coqui TTS framework.

## Setup

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/colabTTS.git
cd colabTTS
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Dataset Structure

The dataset should be organized as follows:
- `train.tsv`: Tab-separated file containing columns for path, sentence, up_votes, and down_votes
- `clips/`: Folder containing all the audio files referenced in train.tsv

## Training Process

1. Open `train_tts.ipynb` in Google Colab
2. Follow the notebook instructions to:
   - Mount your Google Drive
   - Clone this repository
   - Install requirements
   - Copy your dataset
   - Preprocess the data
   - Train the model

## Files Description

- `preprocess.py`: Script for preprocessing audio files and creating metadata
- `train.py`: Main training script using Tacotron2 model
- `train_tts.ipynb`: Colab notebook for running the training process
- `requirements.txt`: List of required Python packages

## Model Architecture

This implementation uses Tacotron2, a state-of-the-art TTS model. The model is configured with:
- Batch size: 32
- Learning rate: Adaptive (using optimizer defaults)
- Audio sampling rate: 22050 Hz
- Text cleaner: english_cleaners

## Output

The trained model and checkpoints will be saved in the `output` directory.
