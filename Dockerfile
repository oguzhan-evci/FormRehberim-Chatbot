# Resmi Python imajını temel al
FROM python:3.10-slim-bullseye

# Çalışma dizinini ayarla
WORKDIR /app

# Gerekli sistem paketlerini kur (FAISS için)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libblas-dev \
    liblapack-dev \
    git && \
    rm -rf /var/lib/apt/lists/*

# Bağımlılıkları kopyala ve kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face cache dizinini çalışma dizini içindeki yazılabilir bir yere ayarla
# TRANSFORMERS_CACHE yerine HF_HOME kullanıyoruz ve /app/.cache dizinine yönlendiriyoruz.
ENV HF_HOME=/app/.cache
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache # Dizini oluştur ve tam yazma izni ver

# Geri kalan uygulama dosyalarını kopyala
COPY . .

# app.py dosyasını çalıştırmak için Flask'ı kullan.
ENV FLASK_APP=app.py
CMD ["flask", "run", "--host", "0.0.0.0", "--port", "7860"]