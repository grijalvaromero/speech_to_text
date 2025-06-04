# Usa imagen oficial de Python
FROM python:3.11-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos
COPY . /app

# Instala las dependencias
RUN apt-get update && apt-get install -y git
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download es_core_news_md

# Expón el puerto (ajusta si usas otro)
EXPOSE 8080

# Comando de ejecución
CMD ["python", "app.py"]
