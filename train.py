import os
import torch
from TTS.trainer import Trainer, TrainerArgs
from TTS.utils.audio import AudioProcessor
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

def train_tts_model(metadata_path, output_path):
    """
    Train a VITS TTS model using the preprocessed dataset
    """
    # Create output path
    os.makedirs(output_path, exist_ok=True)
    
    # Dataset config
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train=metadata_path,
        path=os.path.dirname(metadata_path)
    )

    # Audio config
    audio_config = {
        "sample_rate": 22050,
        "win_length": 1024,
        "hop_length": 256,
        "num_mels": 80,
        "mel_fmin": 0,
        "mel_fmax": None
    }

    # Model config
    config = VitsConfig(
        audio=audio_config,
        run_name="vits_turkish",
        batch_size=32,
        eval_batch_size=16,
        batch_group_size=5,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="turkish_cleaners",
        use_phonemes=True,
        phoneme_language="tr",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config]
    )

    # Init audio processor
    ap = AudioProcessor.init_from_config(config)

    # Init model
    model = Vits(config, ap)

    # Init the trainer and 
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=dataset_config,
        training_assets={"audio_processor": ap},
    )
    trainer.fit()

if __name__ == "__main__":
    metadata_path = "/content/drive/MyDrive/turkish_tts_data/preprocessed_data/metadata.csv"
    output_path = "/content/drive/MyDrive/turkish_tts_data/output"
    train_tts_model(metadata_path, output_path)
