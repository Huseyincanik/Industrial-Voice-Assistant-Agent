import os
import json
import pyodbc
from contextlib import redirect_stdout, redirect_stderr
import time
import keyboard
import re
import tempfile  ### YENİ EKLENDİ ###
import whisper   ### YENİ EKLENDİ ###

import speech_recognition as sr # STT (Mikrofon dinleme) için hala gerekli
from gtts import gTTS             # TTS için eklendi
import pygame                     # Ses çalmak için eklendi

# --- LangChain ve Agent Kütüphaneleri ---
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# --- RAG Bileşenleri ---
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue


# ==============================================================================
# --- 1. KONFİGÜRASYON AYARLARI ---
# ==============================================================================
RAG_BASE_PATH = r""
QDRANT_DB_PATH = os.path.join(RAG_BASE_PATH, "qdrant_db")
QDRANT_COLLECTION_NAME = "machine_manuals"
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large'
LOCAL_LLM_BASE_URL = ""
LOCAL_LLM_API_KEY = "not-needed"
LOCAL_LLM_MODEL_ID = ""
SIMILARITY_THRESHOLD = 0.70
DB_SERVER = r''
DB_DATABASE = ''
DB_USERNAME = ''
DB_PASSWORD = '' 
DB_CONNECTION_STRING = (f'DRIVER={{ODBC Driver 17 for SQL Server}};'f'SERVER={DB_SERVER};'f'DATABASE={DB_DATABASE};'f'UID={DB_USERNAME};'f'PWD={DB_PASSWORD};')

active_machine_id= 1

# ==============================================================================
# --- YENİ SES FONKSİYONLARI ---
# ==============================================================================

def clean_text_for_tts(text):
    """
    TTS için metni istenmeyen özel karakterlerden (örn: *, #, @, _) temizler.
    Bu karakterler gTTS tarafından "yıldız", "kare" vb. olarak okunur.
    """
    # Okunmasını istemediğiniz karakterleri bu köşeli parantez [ ] içine ekleyin.
    unwanted_chars_pattern = r'[#*@_&%]'
    
    # İstenmeyen karakterleri bul ve bir boşluk ' ' ile değiştir.
    clean = re.sub(unwanted_chars_pattern, ' ', text)
    
    # (Opsiyonel) Peş peşe gelen birden fazla boşluğu tek boşluğa indir
    clean = re.sub(r'\s+', ' ', clean).strip()
    
    return clean

# --- GÜNCELLENMİŞ 'speak' FONKSİYONU ---
def speak(text):
    """Verilen metni gTTS ile MP3'e çevirir ve pygame ile sesli olarak okur."""
    
    # Konsola orijinal, temizlenmemiş metni yazdır
    print(f"🤖 Agent: {text}")
    
    # --- ÇÖZÜM ---
    # Metni gTTS'e göndermeden önce özel karakterlerden temizle
    cleaned_text = clean_text_for_tts(text)
    # ---------------
    
    try:
        # gTTS'e 'text' yerine 'cleaned_text'i ver
        tts = gTTS(text=cleaned_text, lang='tr', slow=False)
        
        filename = "response.mp3"
        tts.save(filename)
        
        # Pygame mixer'ın meşgul olmadığından emin ol
        # (Eğer bir önceki ses hala çalıyorsa diye kısa bir kontrol)
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)
            
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        # Çalma sırasında Space tuşuna basılırsa çalmayı kes
        try:
            while pygame.mixer.music.get_busy():
                # Eğer Space tuşuna basıldıysa müziği durdur ve döngüden çık
                if keyboard.is_pressed('space'):
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.05)
        except Exception:
            # keyboard modülü bazı ortamlarda sorun çıkarabilir; burada sessizce devam et
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

        # Dosyayı bırak (unload) ve sil
        try:
            pygame.mixer.music.unload()
        except Exception:
            pass
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except PermissionError:
            print(f"❌ {filename} silinemedi, dosya kullanımda olabilir.")
        
    except Exception as e:
        print(f"❌ Sesli okuma sırasında hata: {e}")
        # Hata durumunda dosyayı silmeyi dene (eğer kaldıysa)
        if 'filename' in locals() and os.path.exists(filename):
            try:
                os.remove(filename)
            except PermissionError:
                print(f"❌ {filename} silinemedi, dosya kullanımda olabilir.")

# ### DEĞİŞTİRİLDİ ###
# --- WHISPER (OFFLINE) KULLANACAK ŞEKİLDE GÜNCELLENMİŞ 'listen_for_command' FONKSİYONU ---
def listen_for_command():
    """'V' tuşuna basılı tutulduğunda mikrofonu dinler ve Whisper ile OFFLINE olarak konuşmayı metne çevirir."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎙️  Konuşmak için 'V' tuşuna basılı tutun...")
        
        # 'V' tuşuna basılmasını bekle
        keyboard.wait('v')
        
        print("🔴 Kaydediliyor... (Konuşmanız bitince tuşu bırakabilirsiniz)")
        
        r.adjust_for_ambient_noise(source, duration=0.5) 
        
        try:
            # 'V' tuşu basılıyken dinle
            audio = r.listen(source, timeout=5, phrase_time_limit=15)
        except sr.WaitTimeoutError:
            speak("Bir şey söylemediğinizi varsayıyorum.")
            return ""

    # --- Google Arama kısmı Whisper (Offline) ile değiştirildi ---
    temp_filepath = None # Hata durumunda silmek için
    try:
        # speech_recognition'dan gelen sesi WAV formatında al
        wav_data = audio.get_wav_data()
        
        # Geçici bir WAV dosyası oluştur
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            temp_audio_file.write(wav_data)
            temp_filepath = temp_audio_file.name

        # Whisper ile çeviri yap (offline)
        # whisper_model global olarak Bölüm 2'de yüklendi
        # fp16=False, CPU uyumluluğu için daha stabildir.
        result = whisper_model.transcribe(temp_filepath, language="tr", fp16=False) 
        
        os.remove(temp_filepath) # Geçici dosyayı sil
        temp_filepath = None

        command = result["text"].strip() # Metni al ve boşlukları temizle

        if not command:
            # Whisper sesi anladı ama boş metin döndürdü
            print("❌ Anlaşılamadı (Whisper boş metin döndürdü).")
            return ""

        print(f"👤 Siz dediniz ki: {command}")
        return command.lower()
        
    except Exception as e:
        # Whisper veya dosya işlemleri sırasında bir hata oluşursa
        print(f"❌ Whisper STT hatası: {e}")
        speak("Sesinizi çevirirken bir hata oluştu.")
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath) # Hata durumunda dosyayı temizle
            except Exception as del_e:
                print(f"❌ Geçici ses dosyası silinirken hata: {del_e}")
        return ""
    # --- Değişiklik sonu ---


# ==============================================================================
# --- 2. AGENT BİLEŞENLERİNİ BİR KERE YÜKLEME ---
# ==============================================================================
print("Embedding modeli (e5-large) yükleniyor, lütfen bekleyin...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
qdrant_client = qdrant_client.QdrantClient(path=QDRANT_DB_PATH)

### DEĞİŞTİRİLDİ ###
# Whisper STT modelini (large-v3) yerel yoldan yükle
WHISPER_MODEL_PATH = os.path.join("models", "large-v3.pt")

print(f"Whisper STT modeli (large-v3) yerel yoldan ({WHISPER_MODEL_PATH}) yükleniyor...")

# Modelin varlığını kontrol et
if not os.path.exists(WHISPER_MODEL_PATH):
    print(f"HATA: Whisper model dosyası bulunamadı: {WHISPER_MODEL_PATH}")
    print("Lütfen önce 'download_model.py' script'ini çalıştırarak modeli indirin.")
    # Model yoksa programdan çık
    exit() 

# Modeli 'large-v3' adı yerine doğrudan dosya yolundan yükle
try:
    whisper_model = whisper.load_model(WHISPER_MODEL_PATH)
    print("✅ Whisper modeli (large-v3) başarıyla yüklendi.")
except Exception as e:
    print(f"❌ Whisper modeli yüklenirken hata oluştu: {e}")
    exit()

#pygame.mixer.init()
with redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
    pygame.mixer.init()

print("✅ Tüm bileşenler başarıyla yüklendi.")


# ==============================================================================
# --- 3. ARAÇ (TOOL) FONKSİYONLARI ---
# ==============================================================================

@tool
def search_specific_machine_documents(query: str, machine_name: str) -> str:
    """Belirli bir makine adı ve zenginleştirilmiş sorgu ile SADECE o makinenin dokümanlarında anlamsal arama yapar ve ilgili metin parçalarını döndürür."""
    print(f"\n>>> DOKÜMAN ARAMA: Makine='{machine_name}', Zenginleştirilmiş Soru='{query}'")
    try:
        query_vector = embedding_model.encode(query).tolist()
        
        
        search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_vector, 
            query_filter=Filter(
                must=[FieldCondition(key="machine_name", match=MatchValue(value=machine_name))]
            ),
            limit=3,
            with_payload=True,
            with_vectors=False
        ) # .search metodu doğrudan sonuç listesini döndürür.
        
        if not search_result:
            return f"'{machine_name}' makinesi için '{query}' sorgusuyla ilgili hiçbir doküman bulunamadı."

        high_quality_results = [result for result in search_result if result.score >= SIMILARITY_THRESHOLD]
        if not high_quality_results:
            highest_score = search_result[0].score if search_result else 0.0
            return (f"'{machine_name}' dokümanlarında konuyla ilgili bölümler arandı ancak yeterince "
                    f"yüksek benzerlikte bir sonuç bulunamadı. Bulunan en yakın sonucun benzerlik skoru ({highest_score:.2f}) "
                    f"belirlenen eşik olan {SIMILARITY_THRESHOLD}'den düşüktür.")

        context_parts = [result.payload['text'] for result in high_quality_results]
        scores = [f"{result.score:.2f}" for result in high_quality_results]
        context = "\n---\n".join(context_parts)
        score_info = f"(Benzerlik Skorları: {', '.join(scores)})"
        return f"'{machine_name}' makinesi için bulunan bilgiler {score_info}:\n{context}"
    except Exception as e:
        return f"Doküman arama sırasında bir hata oluştu: {str(e)}"


# --- SQL ARACI VE BİLEŞENLERİ ---

sql_schema_prompt = """
Sen, endüstriyel makine verileri konusunda uzman bir MS SQL veri analistisin. Görevin, kullanıcının doğal dilde sorduğu soruyu, aşağıdaki şema ve kurallara uygun, çalıştırılabilir tek bir MS SQL sorgusuna çevirmektir. Sadece SQL sorgusunu döndür, başka hiçbir açıklama ekleme.

**VERİTABANI ŞEMASI VE İŞ MANTIĞI:**

1.  **Tablo: `dbo.LogsTable`**
    * **Kullanım Amacı:** Makinelerde meydana gelen genel HATA ve ALARMLARI kaydeder. **'hata', 'alarm', 'log kaydı'** kelimeleri geçtiğinde bu tabloyu kullan.
    * **Sütunlar:** `ID`(int), `LogType`(varchar), `MachineID`(varchar), `ExceptionMessage`(varchar), `CreatedTime`(datetime).

2.  **Tablo: `dbo.AnomalyLogs`**
    * **Kullanım Amacı:** Makine parametrelerinde normal çalışma aralığının dışına çıkan SINIR AŞIMLARINI ve SAPMA DEĞERLERİNİ kaydeder. **'sınır aşımı', 'sapma', 'anomali'** kelimeleri geçtiğinde bu tabloyu kullan.
    * **Sütunlar:** `ID`(int), `MachineId`(varchar), `ExceptionMessage`(varchar), `CreatedTime`(datetime).


3.  **Tablo: `dbo.ComponentData`**
    * **Kullanım Amacı:** Makinelerin 'bıçak ömrü' gibi belirli bileşenlerinin kalan ömrünü veya sayısal değerlerini kaydeder. **'kalan ömür', 'kalan değer', 'bıçak ömrü', 'değeri kaç'** gibi kelimeler geçtiğinde bu tabloyu kullan.
    * **Sütunlar:** `MachineID`(int), `ComponentTypeID`(int), `ComponentVariableID`(int), `MeasuredValue`(float), `CreatedAt`(datetime).
    * **Önemli Mantık:** Bu tablodaki en önemli mantık, `ComponentTypeID` ve `ComponentVariableID` sütunlarının belirli bir ölçümü (`bıçak ömrü` gibi) temsil etmesidir. Bu ID'ler makineden makineye değişebilir. Sorgu her zaman en güncel değeri getirmelidir (`ORDER BY CreatedAt DESC`).
    * **ComponentTypeID ve ComponentVariableID Örnek Eşleştirmeleri:**
        - `ComponentTypeID = 1` ve `ComponentVariableID = 3`: Makine 2'nin bıçak ömrü.
        - `ComponentTypeID = 13` ve `ComponentVariableID = 3`: Makine 1'in bıçak ömrü.
        
**SORGULAMA KURALLARI:**
* **EN ÖNEMLİ KURAL:** `dbo.LogsTable` sorgulanıyorsa, sorguda **MUTLAKA** `WHERE LogType = 'Border'` koşulu bulunmalıdır.
* Tarih belirtilmemişse veya 'en son' deniyorsa en güncel kayıtları getirmek için `TOP 1` (veya istenen sayı kadar) ve `ORDER BY CreatedTime DESC` veya `ORDER BY CreatedAt DESC` kullan.
* `SELECT *` kullanma, sadece ilgili sütunları seç.

**ÖRNEK SORGULAR:**
* **Soru:** 'makine 5 için son sınır aşımı neydi?'
  **SQL:** SELECT TOP 1 MachineId, ExceptionMessage, CreatedTime FROM dbo.AnomalyLogs WHERE MachineId = '5' ORDER BY CreatedTime DESC

* **Soru:** 'en son 3 hata kaydını göster'
  **SQL:** SELECT TOP 3 MachineID, ExceptionMessage, CreatedTime FROM dbo.LogsTable WHERE LogType = 'Border' ORDER BY CreatedTime DESC

* **Soru:** 'makine 1 için bıçağın kalan ömrü nedir?'
  **SQL:** SELECT TOP 1 MeasuredValue FROM dbo.ComponentData WHERE MachineID = 1 AND ComponentTypeID = 13 AND ComponentVariableID = 3 ORDER BY CreatedAt DESC

* **Soru:** 'makine 2'nin bıçak ömrü ne kadar kalmış?'
  **SQL:** SELECT TOP 1 MeasuredValue FROM dbo.ComponentData WHERE MachineID = 2 AND ComponentTypeID = 1 AND ComponentVariableID = 3 ORDER BY CreatedAt DESC


Şimdi, bu bilgilere dayanarak aşağıdaki kullanıcı sorusu için SQL sorgusunu oluştur:
"""

text_to_sql_llm = ChatOpenAI(base_url=LOCAL_LLM_BASE_URL, api_key=LOCAL_LLM_API_KEY, model=LOCAL_LLM_MODEL_ID, temperature=0.1)

@tool
def query_database_for_machine_logs(natural_language_query: str) -> str:
    """Makine hataları, alarmlar, log kayıtları gibi verileri MS SQL veritabanından sorgulamak için kullanılır."""
    
    print(f"\n>>> VERİTABANI (SQL) ARACI KULLANILIYOR: '{natural_language_query}'")
    try:
        response = text_to_sql_llm.invoke([{"role": "system", "content": sql_schema_prompt}, {"role": "user", "content": natural_language_query}])
        sql_query = response.content.strip().replace('`', '').replace('sql', '').strip()
        print(f"     Oluşturulan SQL Sorgusu: {sql_query}")
        if not sql_query: return "Oluşturulan SQL sorgusu boş, işlem yapılamadı."
        with pyodbc.connect(DB_CONNECTION_STRING, timeout=10) as conn:
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            if not rows: return "Veritabanında bu sorguya uygun kayıt bulunamadı."
            columns = [column[0] for column in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]
            return json.dumps(results, indent=2, default=str)
    except Exception as e:
        return f"Veritabanı sorgusu çalıştırılırken hata oluştu. Hata: {e}"


# ==============================================================================
# --- 4. LANGCHAIN AGENT OLUŞTURMA ---
# ==============================================================================
def create_sql_agent():
    print("\nSQL Agent oluşturuluyor...")
    model = ChatOpenAI(base_url=LOCAL_LLM_BASE_URL, api_key=LOCAL_LLM_API_KEY, model=LOCAL_LLM_MODEL_ID, temperature=0.3)
    tools = [query_database_for_machine_logs]
    memory = MemorySaver()
    agent_executor = create_react_agent(model, tools, checkpointer=memory)
    print("✅ LangGraph SQL Agent başarıyla oluşturuldu!")
    return agent_executor


# ==============================================================================
# --- 5. YARDIMCI BİLEŞENLER VE ZİNCİRLER ---
# (Değişiklik yok)
# ==============================================================================

TURKISH_WORDS_TO_NUMS = {
    "bir": "1",
    "iki": "2",
    "üç": "3",
    "dört": "4",
    "beş": "5",
    "altı": "6",
    "yedi": "7",
    "sekiz": "8",
    "dokuz": "9"
}

def convert_word_to_digit(text: str) -> str:
    """
    Kullanıcı 'beş' gibi bir kelime söylerse, bunu '5' gibi bir rakama çevirir.
    Eğer eşleşme bulamazsa, metnin aslını (örn: "5" veya "makine adı") döndürür.
    """
    # Gelen metni (örn: "beş") sözlükte ara.
    # Bulursa, "5" değerini döndür.
    # Bulamazsa, metnin aslını (text) döndür.

    normalized = re.sub(r'\.', '', text.lower())  # Noktalama işaretlerini kaldır ve boşlukları temizle    
    return TURKISH_WORDS_TO_NUMS.get(normalized, text)



def get_machine_list_from_db():
    try:
        response = qdrant_client.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=1000, with_payload=["machine_name"], with_vectors=False)[0]
        if not response: return []
        machine_names = set(point.payload["machine_name"] for point in response if point.payload and "machine_name" in point.payload)
        return sorted(list(machine_names))
    except Exception:
        return []

# Niyet sınıflandırma zinciri
intent_classifier_llm = ChatOpenAI(base_url=LOCAL_LLM_BASE_URL, api_key=LOCAL_LLM_API_KEY, model=LOCAL_LLM_MODEL_ID, temperature=0)

# GÜNCELLENMİŞ PROMPT: Daha net ve sert komutlar içeriyor
intent_prompt = ChatPromptTemplate.from_messages([
    ("system", """SENİN TEK GÖREVİN, kullanıcının sorusunu 'RAG' ya da 'SQL' olarak sınıflandırmaktır.
'RAG', teknik dokümanlar, 'nasıl yapılır' ve bakım prosedürleri gibi genel bilgi soruları içindir.
'SQL', loglar, hatalar, kalan ömür, kalan ... gibi sorular ve alarmlar gibi spesifik veritabanı sorguları içindir.
Cevabın SADECE 'RAG' ya da SADECE 'SQL' olmalıdır. ASLA açıklama yapma, nedenini anlatma veya başka bir kelime ekleme."""),
    ("user", "Kullanıcı Sorusu: {query}")
])
intent_classifier_chain = intent_prompt | intent_classifier_llm | StrOutputParser()

# DÜZELTİLMİŞ SORGULAMA ZİNCİRİ
query_rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", """Sen, teknik dokümanlar konusunda uzman bir mühendissin.
Görevin, kullanıcının sorusunu, bu dokümanların İÇİNDE bulunabilecek anahtar kelimeler ve teknik terimler içeren bir arama sorgusuna dönüştürmektir.
ASLA 'indir', 'pdf', 'ücretsiz' gibi internet aramasına yönelik kelimeler ekleme. Sadece teknik terimlere odaklan.

Örnek:
Orijinal Soru: 'makine bakımı ne zaman yapılır?'
Zenginleştirilmiş Sorgu: 'periyodik bakım tablosu, haftalık bakım prosedürleri, aylık bakım takvimi, önleyici bakım listesi'"""),
    ("user", "Orijinal Soru: {query}\nZenginleştirilmiş Sorgu:")
])
query_rewriter_chain = query_rewriter_prompt | intent_classifier_llm | StrOutputParser()

# Nihai cevap üretme zinciri 
final_response_llm = ChatOpenAI(base_url=LOCAL_LLM_BASE_URL, api_key=LOCAL_LLM_API_KEY, model=LOCAL_LLM_MODEL_ID, temperature=0.7)
final_response_prompt = ChatPromptTemplate.from_messages([
    ("system", "Aşağıdaki bağlamı kullanarak kullanıcının sorusuna net ve anlaşılır bir cevap ver."),
    ("user", "Bağlam:\n{tool_output}\n\nSoru: {query}\n\nCevap:")
])
final_response_chain = final_response_prompt | final_response_llm | StrOutputParser()



# ==============================================================================
# --- 6. ANA UYGULAMA DÖNGÜSÜ ---
# ==============================================================================
def main():
    """
    Kullanıcı ile sesli veya yazılı etkileşime giren, RAG ve SQL mantığını ayıran
    ve global makine ID'si durumunu yöneten ana fonksiyon.
    """
    global active_machine_id # Dışarıdaki 'active_machine_id' değişkenini kullanacağımızı belirtiyoruz

    sql_agent_executor = create_sql_agent()
    config = {"configurable": {"thread_id": "industrial-sql-thread-v3"}}
    
    # Başlangıç mesajı
    initial_greeting = "Merhaba, ben Endüstriyel Agent. Komutlarınızı dinliyorum. 'Yardım' diyerek komut listesini alabilirsiniz."
    speak(initial_greeting)
    
    while True:
        # Komut istemini aktif makine durumuna göre göster
        if active_machine_id:
            print(f"\n--- (Aktif Makine: {active_machine_id}) ---")
        else:
            print("\n--- (Aktif Makine: Seçilmedi) ---")
            
        original_user_input = listen_for_command()
        
        if not original_user_input: 
            continue
        
        # --- Komutları Yönetme ---
        if any(word in original_user_input for word in ["exit", "quit", "çıkış", "kapat"]):
            speak("Görüşmek üzere!")
            break
        
        elif original_user_input.lower() == "yardım":
            help_text = """Kullanılabilir komutlar şunlardır:
            1. Makine seçmek için: 'makine seç' ve ardından makine numarasını söyleyin. Örneğin, 'makine seç 5'.
            2. Aktif makineyi öğrenmek için: 'durum'.
            3. Aktif makine seçimini temizlemek için: 'temizle'.
            4. Çıkmak için: 'çıkış'.
            Bunların dışında doğrudan sorunuzu sorabilirsiniz."""
            speak(help_text)
            print(help_text.replace("            ", "")) # Konsola da düzgün yazdır
            continue

        elif original_user_input.lower().startswith("makine seç"):
            try:
                parts = original_user_input.split()
                # 'makine', 'seç', 'beş' -> son kelimeyi al
                selected_id_str = parts[-1] # Örn: "beş"
                
                # --- YENİ EKLENDİ ---
                # Gelen ID'yi (örn: "beş") rakama (örn: "5") çevirmeyi dene
                processed_id_str = convert_word_to_digit(selected_id_str) # Örn: "5"
                # --- YENİ EKLENDİ SONU ---

                try:
                    # selected_id = int(selected_id_str) # ESKİ KOD
                    selected_id = int(processed_id_str)  # YENİ KOD
                except ValueError:
                    selected_id = -1 
                    # Hata mesajında orijinal, anlaşılamayan kelimeyi kullan
                    speak(f"'{selected_id_str}' anlaşılamadı. Lütfen 'makine seç 5' gibi sayısal bir komut kullanın.")

                
                if selected_id != -1:
                    active_machine_id = str(selected_id) # ID'yi string olarak saklayalım
                    speak(f"Tamamdır, aktif makine {active_machine_id} olarak ayarlandı.")
                
            except (IndexError, ValueError):
                speak("Hatalı komut. Lütfen 'makine seç' ve ardından bir numara söyleyin.")
            continue 
            
        elif original_user_input.lower() == "durum":
            status_text = f"Mevcut aktif makine: {active_machine_id if active_machine_id else 'Seçilmedi'}"
            speak(status_text)
            continue

        elif original_user_input.lower() == "temizle":
            active_machine_id = None
            speak("Aktif makine seçimi temizlendi.")
            continue

        # --- Normal Soru-Cevap Akışı ---
        print("🤖 Niyet anlaşılıyor...", end="", flush=True)
        intent = intent_classifier_chain.invoke({"query": original_user_input})
        intent_clean = intent.strip()
        print(f"\r🤖 Niyet anlaşıldı: {intent_clean}  ")

        try:
            # RAG SÜRECİ
            if "RAG" in intent_clean:
                machine_list = get_machine_list_from_db()
                if not machine_list:
                    speak("Veritabanında hiç makine dokümanı bulunamadı."); continue

                speak("Lütfen aşağıdaki makinelerden birinin adını veya numarasını söyleyin:")
                machine_map = {str(i+1): name for i, name in enumerate(machine_list)}
                machine_map.update({name.lower(): name for name in machine_list}) # İsimle de seçebilsin
                
                for i, name in enumerate(machine_list): 
                    print(f"  {i+1}. {name}")

                while True:
                    choice_str = listen_for_command() # Örn: "beş"
                    if not choice_str: continue
                    
                   
                    processed_choice = convert_word_to_digit(choice_str)
                    

                    # selected_machine = machine_map.get(choice_str.lower()) # ESKİ KOD
                    selected_machine = machine_map.get(processed_choice) # YENİ KOD: İşlenmiş metinle ara
                    
                    if selected_machine:
                        speak(f"Tamam, '{selected_machine}' için arama yapıyorum.")
                        
                        print("🤖 Arama sorgusu zenginleştiriliyor...", end="", flush=True)
                        rewritten_query = query_rewriter_chain.invoke({"query": original_user_input})
                        print(f"\r🤖 Zenginleştirilmiş Sorgu: '{rewritten_query}'  ")

                        context = search_specific_machine_documents.invoke({"query": rewritten_query, "machine_name": selected_machine})
                        
                        speak("Bilgiler alındı, nihai cevap oluşturuluyor.")
                        final_answer = final_response_chain.invoke({"query": original_user_input, "tool_output": context})
                        speak(final_answer)
                        break 
                    else:
                        # Hata mesajında orijinal duyulanı göster
                        speak(f"'{choice_str}' geçersiz bir seçim. Lütfen listeden bir isim veya numara tekrar edin.")
            
            # SQL SÜRECİ
            else:
                if not active_machine_id:
                    speak("SQL sorgusu için lütfen önce bir makine seçin. Örneğin 'makine seç 5' diyebilirsiniz.")
                    continue
                
                print("🤖 SQL Agent düşünüyor...", end="", flush=True)
                sql_input_for_agent = f"Aktif Makine ID'si: {active_machine_id}. Kullanıcı Sorusu: {original_user_input}"
                
                response = sql_agent_executor.invoke({"messages": [{"role": "user", "content": sql_input_for_agent}]}, config)
                final_response = response["messages"][-1].content
                speak(final_response)

        except Exception as e:
            error_message = f"Beklenmedik bir hata oluştu."
            print(f"\n❌ {error_message} Detay: {e}")
            speak("Üzgünüm, beklenmedik bir hata oluştu. Lütfen tekrar deneyin.")

if __name__ == "__main__":

    main()
