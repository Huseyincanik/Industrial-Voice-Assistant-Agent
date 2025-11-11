import os
import whisper

# Modelin indirileceği ve yükleneceği klasör
MODEL_DIR = "models"
MODEL_NAME = "large"  # Çok dilli model
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pt")

def download_model_safely():
    """
    Whisper kütüphanesinin kendi indiricisini kullanarak
    modeli 'models' klasörüne indirir.
    """
    
    # 'models' klasörü yoksa oluştur
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Model zaten indirilmiş mi diye kontrol et
    if os.path.exists(MODEL_PATH):
        print(f"Model dosyası zaten mevcut: {MODEL_PATH}")
        print("İndirme işlemi atlandı. Doğrudan ana script'i çalıştırabilirsiniz.")
        return

    # 2. Model mevcut değilse, whisper'a indirmesini söyle
    print(f"Whisper 'medium' (çok dilli) modeli '{MODEL_DIR}' klasörüne indiriliyor...")
    print("Bu işlem modelin boyutuna ve internet hızınıza bağlı olarak zaman alabilir.")
    
    try:
        # Bu fonksiyon, modeli 'download_root' olarak belirtilen yere indirir
        # ve ardından modeli yükler. Biz sadece indirme kısmı için kullanıyoruz.
        whisper.load_model(MODEL_NAME, download_root=MODEL_DIR)
        
        print(f"\nModel başarıyla indirildi ve doğrulandı.")
        print(f"Dosya konumu: {MODEL_PATH}")

    except Exception as e:
        print(f"\nModel indirilirken bir hata oluştu: {e}")
        print("Lütfen internet bağlantınızı ve 'openai-whisper' kütüphanesinin güncel olduğunu kontrol edin.")

if __name__ == "__main__":
    download_model_safely()