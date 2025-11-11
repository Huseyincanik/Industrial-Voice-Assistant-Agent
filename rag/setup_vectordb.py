# Bu betiğin amacı, pipeline'ı çalıştırarak elde edilen chunk ve embedding'leri
# kalıcı bir Qdrant veritabanına YALNIZCA BİR KEZ yüklemektir.
# Bu betik çalıştırıldıktan sonra, soru-cevap uygulaması doğrudan
# diske kaydedilmiş bu veritabanını kullanacaktır.

from rag_pipeline import load_documents, chunk_texts, generate_embeddings
import qdrant_client
from qdrant_client.http.models import PointStruct, UpdateStatus, Distance, VectorParams
import numpy as np
from typing import List, Any

# --- Konfigürasyon Ayarları ---
# rag_pipeline.py dosyasındaki ayarlarla aynı olmalı
DOCUMENTS_PATH = r"C:\Users\PC1\Desktop\get_agent_log\documents"
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large'
QDRANT_DB_PATH = "./qdrant_db"  # Veritabanının kaydedileceği klasör
QDRANT_COLLECTION_NAME = "machine_manuals"
RAG_BASE_PATH = r"C:\Users\PC1\Desktop\get_agent_log\rag"

def create_and_populate_vectordb():
    """
    Tüm süreci yönetir: Veriyi yükler, chunk'lar, embed eder ve Qdrant'a yazar.
    """
    # Adım 1, 2, 3: rag_pipeline.py'daki fonksiyonları kullanarak veriyi bellekte hazırla.
    documents = load_documents(r"C:\Users\PC1\Desktop\get_agent_log\documents")
    if not documents: return

    chunks = chunk_texts(documents) # Bu, artık machine_name içeren chunk'ları döndürecek
    if not chunks: return

    embeddings = generate_embeddings(chunks, EMBEDDING_MODEL_NAME)

    # Adım 4: Veriyi Qdrant'a aktar
    print("\n--- Kalıcı Vektör Veritabanı Kuruluyor ---")
    client = qdrant_client.QdrantClient(path=QDRANT_DB_PATH)
    
    vector_size = embeddings.shape[1]
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"'{QDRANT_COLLECTION_NAME}' koleksiyonu başarıyla oluşturuldu.")

    client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        wait=True,
        points=[
            PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={
                    "text": chunk.page_content,
                    "source": chunk.metadata["source"],
                    "machine_name": chunk.metadata["machine_name"] # ÖNEMLİ KISIM
                }
            ) for i, (chunk, vector) in enumerate(zip(chunks, embeddings))
        ]
    )
    
    # Makine adına göre hızlı filtreleme yapabilmek için payload index'i oluşturuyoruz
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="machine_name",
        field_schema="keyword"
    )
    print("✅ 'machine_name' için payload index'i oluşturuldu.")

    collection_info = client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
    print(f"Doğrulama: Koleksiyondaki toplam nokta sayısı: {collection_info.points_count}")
    print("---------------------------------------------")

if __name__ == "__main__":
    create_and_populate_vectordb()
    print("\nİşlem tamamlandı. 'qdrant_db' klasörü içinde kalıcı veritabanınız yeniden oluşturuldu.")
    print("Artık 'agent4.py' dosyasını çalıştırarak test edebilirsiniz.")