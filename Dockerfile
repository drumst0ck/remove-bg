FROM python:3.9

# Instalar dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Crea el directorio 'static' si no existe
RUN mkdir -p static

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]