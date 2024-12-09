import os
import torch
import json
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.generic_utils import setup_model
from TTS.tts.utils.text.symbols import symbols, phonemes
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.visual import plot_spectrogram
from TTS.tts.datasets import load_meta_data
from TTS.tts.utils.managers import BaseTTSManager

def load_config(config_path):
    """Load config from a json file"""
    with open(config_path, 'r') as f:
        return AttrDict(json.load(f))

class AttrDict(dict):
    """A dictionary that allows for attribute-style access"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

def train_tts_model(metadata_path, output_path):
    """
    Train a TTS model using the preprocessed dataset
    """
    # Create output path
    os.makedirs(output_path, exist_ok=True)
    
    # Load config
    config = load_config("config.json")
    if not hasattr(config, 'audio'):
        config.audio = AttrDict()
    config.audio.sample_rate = 22050
    
    # Init audio processor
    ap = AudioProcessor(**config.audio)
    
    # Load training samples
    train_samples, eval_samples = load_meta_data(metadata_path)
    
    # Init model
    model = setup_model(len(phonemes) if config.use_phonemes else len(symbols), config)
    
    # Init trainer
    trainer = BaseTTSManager(
        model=model,
        config=config,
        output_path=output_path,
        train_samples=train_samples,
        eval_samples=eval_samples,
        training_assets={"audio_processor": ap}
    )
    
    # Train the model
    trainer.fit()

if __name__ == "__main__":
    metadata_path = "/content/drive/MyDrive/turkish_tts_data/preprocessed_data/metadata.csv"
    output_path = "/content/drive/MyDrive/turkish_tts_data/output"
    train_tts_model(metadata_path, output_path)
