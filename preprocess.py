import os
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf

def preprocess_dataset(tsv_path, clips_folder, output_folder):
    """
    Preprocess the dataset by validating audio files and creating a cleaned metadata file.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Read the TSV file
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Initialize lists for valid entries
    valid_paths = []
    valid_sentences = []
    
    print("Preprocessing audio files...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = os.path.join(clips_folder, row['path'])
        
        try:
            # Load and validate audio file
            audio, sr = librosa.load(audio_path, sr=22050)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Save preprocessed audio
            output_path = os.path.join(output_folder, row['path'])
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio, sr)
            
            # Add to valid entries
            valid_paths.append(row['path'])
            valid_sentences.append(row['sentence'])
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            continue
    
    # Create new metadata DataFrame
    cleaned_df = pd.DataFrame({
        'audio_path': valid_paths,
        'text': valid_sentences
    })
    
    # Save cleaned metadata
    metadata_path = os.path.join(output_folder, 'metadata.csv')
    cleaned_df.to_csv(metadata_path, index=False)
    
    print(f"Preprocessing complete. Processed {len(valid_paths)} valid files.")
    return metadata_path

if __name__ == "__main__":
    # Google Drive paths
    tsv_path = "/content/drive/MyDrive/turkish_tts_data/train.tsv"
    clips_folder = "/content/drive/MyDrive/turkish_tts_data/clips"
    output_folder = "/content/drive/MyDrive/turkish_tts_data/preprocessed_data"
    preprocess_dataset(tsv_path, clips_folder, output_folder)
