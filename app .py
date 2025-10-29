import os
import re
import glob
import time
import markdown
import traceback
import google.generativeai as genai

from flask import Flask, render_template, request, session, redirect, url_for, g
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

print("[INIT] app.py dosyası çalışmaya başladı. Kütüphaneler yüklendi.")

# --- 1. Uygulama Başlangıç Ayarları (Setup) ---

# Hugging Face Secrets'tan API anahtarını al
API_KEY = os.environ.get("GEMINI_API_KEY")
print("[INIT] API_KEY 'GEMINI_API_KEY' secret'ından okunmaya çalışıldı.")

if not API_KEY:
    print("[HATA] HATA: GEMINI_API_KEY ortam değişkeni bulunamadı. Lütfen HF Space secrets'a ekleyin.")
else:
    print("[INIT] GEMINI_API_KEY başarıyla bulundu.")

llm = None
retriever = None
qa_chain_with_history = None

try:
    print("[INIT] RAG bileşenleri (LLM, Embeddings, FAISS) try bloğuna giriliyor...")
    if API_KEY:
        genai.configure(api_key=API_KEY)

        # 1.1. LLM Modelini Yükle
        # === DÜZELTME: API_KEY'i doğrudan parametre olarak geç ===
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.85,
            top_p=0.9,
            google_api_key=API_KEY # <-- DÜZELTME BURADA
        )
        print("[INIT] LLM Modeli (gemini-2.5-flash) yapılandırıldı.")
    else:
        print("[UYARI] UYARI: API Anahtarı yok, LLM yüklenemedi.")


    # 1.2. Embedding Modelini Yükle
    print("[INIT] Embedding modeli (all-MiniLM-L6-v2) yükleniyor...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("[INIT] Embedding modeli yüklendi.")

    # 1.3. HAZIR FAISS Veritabanını Yükle
    print("[INIT] FAISS Vektör Veritabanı ('faiss_exercise_index') yükleniyor...")
    vector_store = FAISS.load_local(
        "faiss_exercise_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print("[INIT] FAISS Vektör Veritabanı başarıyla yüklendi ve retriever oluşturuldu.")

except Exception as e:
    print(f"[HATA] HATA: Başlangıçta RAG bileşenleri yüklenirken KRİTİK HATA: {e}")
    print(traceback.format_exc())
    # Hata durumunda bileşenlerin None olduğundan emin ol
    llm = None
    retriever = None

# --- 2. RAG Promptları ve Zincirleri (Kaggle Hücre 6'dan) ---
print("[INIT] RAG Promptları tanımlanıyor...")

# 2.1. Hafıza için Sorgulama Prompt'u
history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Yukarıdaki konuşmaya dayanarak, sadece son soruyu cevaplamak için vektör veritabanında arama yapmaya uygun, tek başına bir sorgu cümlesi oluştur.")
])

# 2.2. Egzersiz Bilgisi / Sohbet Prompt'u (Maksimum Özgünlük)
exercise_info_prompt_template = """**SENİN ROLÜN:** Sen Active Break Egzersiz Asistanısın. Bilgiyi **derinlemesine anlayıp**, sanki bir uzmanın **doğal sohbet tarzıyla** açıklıyormuş gibi, **tamamen kendi özgün ifadelerinle** sunan, pozitif ve net bir rehbersin. Referans metni okuyormuş gibi **duyulmamalısın**.
**ANA GÖREVİN:** Kullanıcının girdisini analiz et ve aşağıdaki süreci izleyerek yanıtını oluştur:
1.  **NİYETİ BELİRLE:** Kullanıcı sohbet mi ediyor, spesifik bir egzersiz mi soruyor, yoksa belirsiz bir istekte mi bulunuyor?
    * **Sohbetse:** Bağlamı (`Referans Bilgiler`) yok say. Kısa, nazik ve **yardıma odaklı** bir yanıt ver. (Örn: "Merhaba! Size hangi egzersiz hakkında bilgi verebilirim?"). -> **Bitir.**
    * **Soru/İstekse:** Devam et.
2.  **REFERANS BİLGİYİ ANALİZ ET (Soru/İstekse):**
    * 'Referans Bilgiler (Bağlam)' senin **TEK BİLGİ KAYNAĞINDIR**.
    * Kullanıcının sorusuyla **doğrudan ilgili** bilgileri **belirle ve ANLA**. Sadece anahtar kelimeleri değil, anlamını kavra.
3.  **ÖZGÜN VE KESİN CEVABINI YARAT (Soru/İstekse - EN KRİTİK ADIM):**
    * **Eğer Alakalı Referans Varsa:**
        * **KESİNLİKLE SADECE SORULANA ODAKLAN:**
          * Kullanıcı "Squat nedir?" diye sorarsa, **sadece tanımını** ver.
          * Kullanıcı "Hangi kasları çalıştırır?" diye sorarsa, **sadece kas listesini** ver.
          * Kullanıcı "Nasıl yapılır?" diye sorarsa, **sadece 'Nasıl Yapılır' adımlarını** ver.
          * Soru belirsizse (örn: "Squat hakkında bilgi ver"), tanım, kaslar ve yapılış hakkında kısa bir özet ver.
        * **MAKSİMUMLU ÖZGÜNLİKLE YENİDEN YAZ (EN ÖNEMLİ KURAL):** Referans Bilgiler'deki ilgili kısmı **özümsedikten sonra**, bu bilgiyi **tamamen kendi kelimelerinle, FARKLI CÜMLE YAPILARI, EŞ ANLAMLILAR ve kendi anlatım tarzınla sıfırdan ifade et**. Referans metindeki **ANAHTAR KELİME GRUPLARINI veya CÜMLE YAPILARINI KULLANMAKTAN KAÇIN**. Cevabın, referans metnin bir kopyası veya hafifçe değiştirilmiş hali gibi **GÖRÜNMEMELİDİR**. Sanki bilgiyi kendin biliyormuş ve doğal bir şekilde anlatıyormuş gibi olmalı.
        * **KISA VE İLGİLİ BİR KAPANIŞ EKLE (İsteğe Bağlı):** Cevabı verdikten sonra, **eğer uygunsa**, cevapla doğrudan ilgili, **çok kısa (tek cümle)** ve doğal bir olumlu ifade ekleyebilirsin. Aşırıya kaçma.
    * **Eğer Alakalı Referans Yoksa VEYA Soru Belirsizse:**
        * **KESİNLİKLE GENEL BİLGİNİ KULLANMA.**
        * **Belirsiz Soruları Yönet:** (Örn: "bacak egzersizleri") "Veri setimde genel 'bacak egzersizleri' listesi yerine spesifik hareketler bulunuyor. Örneğin **Squat** veya **Lunge** hakkında bilgi verebilirim. Hangisini öğrenmek istersin?"
        * **Bilgi Yoksa (Spesifik Soru İçin):** "Üzgünüm, '[Kullanıcının Sorduğu Konu]' hakkında sağlanan bilgilerde detay bulamadım. Belki Squat, Plank veya Lunge gibi temel hareketlerden birini sormak istersin?"
4.  **FORMATLAMA VE ÜSLUP:**
    * Cevabını **basit Markdown** (örn: **kalın**, `- liste`) ile okunaklı yap.
    * Her zaman **pozitif, teşvik edici ama NET ve ÖZ** ol.
    * Tıbbi tavsiye verme. Egzersiz önerme (sadece sorulanı açıkla).
**Referans Bilgiler (Bağlam) (TEK BİLGİ KAYNAĞIN):**
{context}
**Kullanıcının Girdisi:** {input}
**SENİN TAMAMEN ÖZGÜNLEŞTİRİLMİŞ, NET VE ODAKLI CEVABIN:**
"""

exercise_info_prompt = ChatPromptTemplate.from_messages([
    ("system", exercise_info_prompt_template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])

# 2.3. RAG Zincirlerini Oluştur
print("[INIT] RAG zincirleri (Promptlar sonrası) oluşturuluyor...")
if llm and retriever:
    try:
        history_aware_retriever_chain = create_history_aware_retriever(
            llm, retriever, history_aware_retriever_prompt
        )
        answer_generation_chain = create_stuff_documents_chain(llm, exercise_info_prompt)
        qa_chain_with_history = create_retrieval_chain(
            history_aware_retriever_chain, answer_generation_chain
        )
        print("[INIT] Hafızalı RAG zinciri başarıyla oluşturuldu.")
    except Exception as e:
        print(f"[HATA] HATA: RAG zinciri oluşturulurken KRİTİK HATA: {e}")
        qa_chain_with_history = None # Hata durumunda None yap
else:
    print("[UYARI] UYARI: LLM veya Retriever (veya her ikisi) yüklenemediği için RAG zinciri oluşturulamadı.")


# --- 3. FLASK UYGULAMASI (Kaggle Hücre 9'dan) ---

print("[INIT] Flask uygulaması 'app = Flask(...)' oluşturuluyor.")
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'super_secret_key_form_rehberim_v2_exlist_hf' # Yeni key

# Hugging Face iframe cookie ayarları (Dil değiştirme sorunu için)
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True
print("[INIT] Flask cookie ayarları (iframe için) yapıldı.")


# --- Dil Verileri (Değişiklik yok) ---
LANG_DATA = {
    'tr': {
        'title': 'Form Rehberim',
        'chatbot_title': 'Form Rehberim',
        'chatbot_subtitle': 'Hareketlerin nasıl yapıldığını ve inceliklerini sorun.',
        'welcome_message': 'Merhaba! Ben sizin Form Rehberinizim. Squat, Plank, Lunge gibi hareketlerin nasıl yapıldığı, hangi kasları çalıştırdığı gibi konularda bilgi almak için bana spesifik egzersiz adını sorabilirsiniz. Size özel antrenman programları oluşturamam, ancak mevcut egzersizler hakkında detaylı bilgi verebilirim. Hangi hareketi merak ediyorsunuz?',
        'loading_text': 'Cevap Aranıyor...',
        'input_placeholder': 'Örn: Squat nasıl yapılır? / Plank hangi kasları çalıştırır?',
        'send_button': 'Gönder',
        'clear_chat_button': 'Sohbeti Temizle',
        'nav_home': 'Ana Sayfa',
        'nav_exercises': 'Egzersiz Listesi',
        'nav_about': 'Hakkında',
        'about_title': 'Hakkında - Form Rehberim',
        'about_page_heading': 'Form Rehberim Hakkında',
        'about_paragraph_1': 'Bu yapay zeka asistanı, sadece evde veya istediğiniz yerde yapabileceğiniz temel vücut ağırlığı egzersizleri (Squat, Plank, Lunge vb.) hakkında bilgi sağlamak amacıyla tasarlanmıştır. Hareketlerin doğru yapılışı ve temel detayları hakkında sorular sorabilirsiniz.',
        'about_paragraph_2': 'Amacımız, temel egzersizleri doğru formda öğrenmenize yardımcı olarak daha bilinçli hareket etmenizi sağlamaktır. Bu asistan, size özel antrenman programları oluşturmaz veya kişisel fitness tavsiyesi vermez, yalnızca mevcut egzersiz kütüphanesindeki bilgileri sunar.',
        'about_contact_heading': 'Geri Bildirim',
        'about_contact_info': 'Uygulama hakkındaki düşüncelerinizi veya karşılaştığınız sorunları belirtirseniz sevinirim.',
        'back_to_chat': 'Sohbete Geri Dön',
        'error_message': 'Üzgünüm, bir hata oluştu: {error}',
        'chatbot_not_ready_message': 'Chatbot bileşenleri henüz hazır değil (Lütfen logları kontrol edin).',
        'exercise_list_title': 'Egzersiz Listesi',
        'exercise_list_intro': 'Aşağıda hakkında bilgi alabileceğiniz egzersizlerin listesini bulabilirsiniz:'
    },
    'en': {
         'title': 'Form Guide',
         'chatbot_title': 'Form Guide',
         'chatbot_subtitle': 'Ask how exercises are done and their intricacies.',
         'welcome_message': 'Hello! I am your Form Guide Assistant. You can ask me for the name of a specific exercise to get information on topics such as how movements like Squat, Plank, Lunge are performed and which muscles they work. I cannot create personalized training programs for you, but I can provide detailed information about existing exercises. Which movement are you curious about?',
         'loading_text': 'Searching for an answer...',
         'input_placeholder': 'E.g., How to do a Squat? / What muscles does Plank work?',
         'send_button': 'Send',
         'clear_chat_button': 'Clear Chat',
         'nav_home': 'Home',
         'nav_exercises': 'Exercise List',
         'nav_about': 'About',
         'about_title': 'About - Form Guide',
         'about_page_heading': 'About Form Guide',
         'about_paragraph_1': 'This AI assistant is designed only to provide information about basic bodyweight exercises (Squat, Plank, Lunge, etc.) that you can do at home or anywhere. You can ask questions about the correct execution and basic details of the movements.',
         'about_paragraph_2': 'Our aim is to help you perform basic exercises with the correct form, enabling you to move more consciously. This assistant does not create personalized training programs or provide personal fitness advice, it only presents information from the existing exercise library.',
         'about_contact_heading': 'Feedback',
         'about_contact_info': 'I would appreciate it if you could share your thoughts about the application or any issues you encountered.',
         'back_to_chat': 'Back to Chat',
         'error_message': 'Sorry, an error occurred: {error}',
         'chatbot_not_ready_message': 'The chatbot components are not ready yet (Please check logs).',
         'exercise_list_title': 'Exercise List',
         'exercise_list_intro': 'Below you can find the list of exercises you can ask about:'
    }
}
print("[INIT] Dil verileri (LANG_DATA) yüklendi.")

# --- Yardımcı Fonksiyonlar ---
def convert_markdown_to_html(md_text):
    try:
        html = markdown.markdown(md_text, extensions=['fenced_code', 'nl2br'])
        return html
    except Exception as e:
        print(f"Markdown dönüşüm hatası: {e}")
        return md_text

def get_exercise_list():
    # DİKKAT: Kaggle yolu yerine yerel yolu (kopyalanan) kullanıyoruz
    ansiklopedi_dir = "hareket_ansiklopedisi"

    exercise_files = []
    if os.path.exists(ansiklopedi_dir):
        pattern = os.path.join(ansiklopedi_dir, "*.md")
        files = glob.glob(pattern)
        for f_path in files:
            f_name = os.path.basename(f_path)
            exercise_name = os.path.splitext(f_name)[0]
            display_name = exercise_name.replace('-', ' ').replace('_', ' ').title()
            if not display_name[0].isupper():
                 display_name = display_name[0].upper() + display_name[1:]

            exercise_files.append(display_name)
        exercise_files.sort()
    return exercise_files

print("[INIT] Yardımcı fonksiyonlar (convert_markdown, get_exercise_list) tanımlandı.")

# --- Flask Rotaları (Kaggle Hücre 9'dan) ---
@app.before_request
def before_request():
    g.lang = session.get('lang', 'tr')
    g.lang_data = LANG_DATA.get(g.lang, LANG_DATA['tr'])

@app.route('/set_language/<lang_code>')
def set_language(lang_code):
    if lang_code in LANG_DATA:
        session['lang'] = lang_code
    return redirect(request.referrer or url_for('home'))

@app.route('/about')
def about():
    return render_template('about.html', lang=g.lang, lang_data=g.lang_data)

@app.route('/egzersizler')
def exercise_list():
    exercises = get_exercise_list()
    # print(f"{len(exercises)} adet egzersiz listelendi.") # print'i log kirliliği için kapatalım
    return render_template('egzersizler.html', exercises=exercises, lang=g.lang, lang_data=g.lang_data)

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    session.pop('chat_history', None)
    print("Sohbet geçmişi temizlendi.")
    return redirect(url_for('home'))

@app.route('/', methods=['GET', 'POST'])
def home():
    global qa_chain_with_history

    simple_chat_history = session.get('chat_history', [])
    question = ""
    answer_html = ""

    if request.method == 'POST':
        question = request.form.get('question', '').strip()

        if not question:
            return render_template('index.html', chat_history=simple_chat_history, lang=g.lang, lang_data=g.lang_data)

        # 'qa_chain_with_history' None ise (başlangıçta yüklenemediyse)
        if not qa_chain_with_history:
            print("[HATA] KULLANICI SORDU, AMA RAG ZİNCİRİ HAZIR DEĞİL (qa_chain_with_history is None).")
            answer = g.lang_data['chatbot_not_ready_message']
            answer_html = convert_markdown_to_html(answer)
            simple_chat_history.append((question, answer_html))
            session['chat_history'] = simple_chat_history
            return render_template('index.html', chat_history=simple_chat_history, lang=g.lang, lang_data=g.lang_data)

        simple_chat_history.append((question, None))
        session['chat_history'] = simple_chat_history

        try:
            print(f"--- Hafızalı RAG Zinciri Çalıştırılıyor --- Sorgu: '{question}'")
            langchain_chat_history = []
            for q, a_html in simple_chat_history[:-1]:
                 if q and a_html:
                     a_text = re.sub('<[^<]+?>', '', a_html) if a_html else ""
                     langchain_chat_history.append(HumanMessage(content=q))
                     langchain_chat_history.append(AIMessage(content=a_text))

            print(f">>> LLM ÇAĞRISI (HAFIZALI QA) BAŞLIYOR...")
            start_time = time.time()
            result = qa_chain_with_history.invoke({"input": question, "chat_history": langchain_chat_history })
            end_time = time.time()
            print(f"<<< LLM ÇAĞRISI (HAFIZALI QA) BAŞARILI ({end_time - start_time:.2f} saniye).")

            raw_answer = result['answer']
            answer_html = convert_markdown_to_html(raw_answer)

            if simple_chat_history: simple_chat_history[-1] = (question, answer_html)
            session['chat_history'] = simple_chat_history

        except Exception as e:
            print(f"[HATA] HATA: RAG zinciri çalıştırılırken (invoke) hata: {e}")
            print(traceback.format_exc())
            answer = g.lang_data['error_message'].format(error=str(e))
            answer_html = convert_markdown_to_html(answer)
            if simple_chat_history: simple_chat_history[-1] = (question, answer_html)
            session['chat_history'] = simple_chat_history

        return render_template('index.html', chat_history=simple_chat_history, question=question, answer=answer_html, lang=g.lang, lang_data=g.lang_data)

    return render_template('index.html', chat_history=simple_chat_history, lang=g.lang, lang_data=g.lang_data)

print("[INIT] Tüm Flask rotaları (@app.route) tanımlandı.")

# --- 4. Uygulamayı Başlatma (HF Spaces için) ---
print("[INIT] Dosyanın sonu, 'if __name__ == __main__' bloğu tanımlanıyor.")
if __name__ == '__main__':
    # HF Spaces, uygulamayı 7860 portunda bekler
    # 'debug=False' produksiyon için önemlidir.
    print("[INIT] UYGULAMA BAŞLATILIYOR (app.run)...")
    port = int(os.environ.get('PORT', 7860))
    # === NOT: Dockerfile'daki CMD gunicorn kullandığı için app.run aslında burada çağrılmıyor ===
    # === Ancak kodun burada olması Flask'ın geliştirme sunucusuyla test etmeyi kolaylaştırır ===
    # === Gerçek başlatma Dockerfile'daki CMD ile yapılır. ===
    # app.run(host='0.0.0.0', port=port, debug=False) # Gunicorn kullanıldığı için bu satıra gerek yok ama kalsa da zararı olmaz
    pass # Gunicorn zaten app nesnesini bulup çalıştıracak