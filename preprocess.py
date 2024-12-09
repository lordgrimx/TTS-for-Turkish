import os
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
import traceback
import time

def read_tsv_in_chunks(tsv_path, max_samples=1000, chunksize=100):
    """
    TSV dosyasını parça parça okur
    """
    print("\nTSV dosyası parça parça okunuyor...")
    chunks = []
    try:
        for chunk in pd.read_csv(tsv_path, sep='\t', chunksize=chunksize):
            chunks.append(chunk)
            if len(pd.concat(chunks)) >= max_samples:
                break
        return pd.concat(chunks).head(max_samples)
    except Exception as e:
        print(f"TSV dosyası okunurken hata oluştu: {str(e)}")
        print("Hata detayı:")
        print(traceback.format_exc())
        return None

def preprocess_dataset(tsv_path, clips_folder, output_folder, max_samples=1000):
    """
    Preprocess the dataset by validating audio files and creating a cleaned metadata file.
    """
    # Dosya yollarını kontrol et
    print(f"TSV dosyası mevcut mu: {os.path.exists(tsv_path)}")
    print(f"Clips klasörü mevcut mu: {os.path.exists(clips_folder)}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # TSV dosyasını parça parça oku
    df = read_tsv_in_chunks(tsv_path, max_samples)
    if df is None:
        print("TSV dosyası okunamadı. İşlem durduruluyor.")
        return None
    
    # Path sütunundaki yolları düzelt
    df['path'] = df['path'].apply(lambda x: os.path.basename(x))  # Sadece dosya adını al
    
    print(f"TSV dosyası içeriği (ilk 5 satır):\n{df.head()}")
    print(f"İşlenecek toplam satır sayısı: {len(df)}")
    
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
    success_count = 0
    
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
            
            # Dosya boyutunu kontrol et
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                print(f"\nUyarı: {audio_path} dosyası boş (0 byte)")
                error_count += 1
                continue
                
            # Load and validate audio file
            start_time = time.time()
            try:
                audio, sr = librosa.load(audio_path, sr=22050, timeout=30)  # 30 saniyelik timeout ekle
            except Exception as e:
                print(f"\nHata: {audio_path} dosyası yüklenirken zaman aşımı veya hata oluştu:")
                print(f"Hata detayı: {str(e)}")
                print(f"Dosya boyutu: {file_size} bytes")
                error_count += 1
                continue
            
            load_time = time.time() - start_time
            if load_time > 5:  # 5 saniyeden uzun süren yüklemeleri raporla
                print(f"\nUyarı: {audio_path} dosyasının yüklenmesi {load_time:.2f} saniye sürdü")
            
            # Ses dosyasının uzunluğunu kontrol et
            duration = len(audio) / sr
            if duration > 30:  # 30 saniyeden uzun ses dosyalarını atla
                print(f"\nUyarı: {audio_path} dosyası çok uzun ({duration:.2f} saniye)")
                error_count += 1
                continue
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Save preprocessed audio
            output_path = os.path.join(output_folder, row['path'])
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio, sr)
            
            # Add to valid entries
            valid_paths.append(row['path'])
            valid_sentences.append(row['sentence'])
            success_count += 1
            
            # Her 100 başarılı işlemde bir rapor ver
            if success_count % 100 == 0:
                print(f"\nBaşarılı işlem sayısı: {success_count}")
            
        except Exception as e:
            print(f"\nHata: {audio_path} dosyası işlenirken beklenmeyen bir hata oluştu:")
            print(f"Hata detayı: {str(e)}")
            print("Hata izleme:")
            print(traceback.format_exc())
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
    print(f"- İşlenen toplam dosya sayısı: {len(df)}")
    print(f"- Başarıyla işlenen dosya sayısı: {len(valid_paths)}")
    print(f"- Hatalı dosya sayısı: {error_count}")
    return metadata_path

if __name__ == "__main__":
    # Google Drive paths
    tsv_path = "/content/drive/MyDrive/turkish_tts_data/train.tsv"
    clips_folder = "/content/drive/MyDrive/turkish_tts_data/clips"
    output_folder = "/content/drive/MyDrive/turkish_tts_data/preprocessed_data"
    preprocess_dataset(tsv_path, clips_folder, output_folder, max_samples=1000)  # Sadece ilk 1000 veriyi işle
