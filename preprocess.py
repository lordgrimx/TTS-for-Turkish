import os
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf

def preprocess_dataset(tsv_path, clips_folder, output_folder):
    """
    Preprocess the dataset by validating audio files and creating a cleaned metadata file.
    """
    # Dosya yollarını kontrol et
    print(f"TSV dosyası mevcut mu: {os.path.exists(tsv_path)}")
    print(f"Clips klasörü mevcut mu: {os.path.exists(clips_folder)}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # TSV dosyasını oku ve içeriğini kontrol et
    print("\nTSV dosyası okunuyor...")
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"TSV dosyası içeriği (ilk 5 satır):\n{df.head()}")
    print(f"Toplam satır sayısı: {len(df)}")
    
    # İlk ses dosyasının yolunu kontrol et
    if len(df) > 0:
        first_audio = os.path.join(clips_folder, df['path'].iloc[0])
        print(f"\nİlk ses dosyası yolu: {first_audio}")
        print(f"Bu dosya mevcut mu: {os.path.exists(first_audio)}")
        
        # Klasör içeriğini kontrol et
        print(f"\nClips klasörü içeriği:")
        try:
            files = os.listdir(clips_folder)
            print(f"Klasördeki dosya sayısı: {len(files)}")
            if len(files) > 0:
                print(f"İlk birkaç dosya: {files[:5]}")
        except Exception as e:
            print(f"Klasör içeriği okunamadı: {str(e)}")
    
    # Initialize lists for valid entries
    valid_paths = []
    valid_sentences = []
    error_count = 0
    
    print("\nSes dosyaları işleniyor...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = os.path.join(clips_folder, row['path'])
        
        try:
            # Önce dosyanın varlığını kontrol et
            if not os.path.exists(audio_path):
                if error_count < 5:  # Sadece ilk 5 hatayı göster
                    print(f"\nDosya bulunamadı: {audio_path}")
                error_count += 1
                continue
                
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
            if error_count < 5:  # Sadece ilk 5 hatayı göster
                print(f"\nHata: {audio_path} dosyası işlenirken sorun oluştu:")
                print(f"Hata detayı: {str(e)}")
            error_count += 1
            continue
    
    print(f"\nToplam hata sayısı: {error_count}")
    
    # Create new metadata DataFrame
    cleaned_df = pd.DataFrame({
        'audio_path': valid_paths,
        'text': valid_sentences
    })
    
    # Save cleaned metadata
    metadata_path = os.path.join(output_folder, 'metadata.csv')
    cleaned_df.to_csv(metadata_path, index=False)
    
    print(f"\nİşlem tamamlandı:")
    print(f"- Toplam dosya sayısı: {len(df)}")
    print(f"- Başarıyla işlenen dosya sayısı: {len(valid_paths)}")
    print(f"- Hatalı dosya sayısı: {error_count}")
    return metadata_path

if __name__ == "__main__":
    # Google Drive paths
    tsv_path = "/content/drive/MyDrive/turkish_tts_data/train.tsv"
    clips_folder = "/content/drive/MyDrive/turkish_tts_data/clips"
    output_folder = "/content/drive/MyDrive/turkish_tts_data/preprocessed_data"
    preprocess_dataset(tsv_path, clips_folder, output_folder)
