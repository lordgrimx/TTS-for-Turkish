import os
import torch
from TTS.config import load_config
from TTS.trainer import Trainer, TrainingArgs
from TTS.utils.audio import AudioProcessor
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

def train_tts_model(metadata_path, output_path):
    """
    Train a Tacotron2 TTS model using the preprocessed dataset
    """
    # Create output path
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize config
    config = Tacotron2Config(
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="english_cleaners",
        use_phonemes=False,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[{"name": "custom", "path": metadata_path}]
    )

    # Initialize audio processor
    ap = AudioProcessor.init_from_config(config)

    # Load training samples
    train_samples, eval_samples = load_tts_samples(
        dataset_config=config.datasets[0],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # Init model
    model = Tacotron2(config)

    # Init the trainer and 🚀
    trainer = Trainer(
        TrainingArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        training_assets={"audio_processor": ap},
    )
    trainer.fit()

if __name__ == "__main__":
    metadata_path = "preprocessed_data/metadata.csv"
    output_path = "output"
    train_tts_model(metadata_path, output_path)
