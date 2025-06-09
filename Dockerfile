# Usa imagen oficial de Python
FROM python:3.11

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos
COPY . /app

# Instala git y otras dependencias necesarias
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Instala requerimientos
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache /tmp/*


# Descarga el modelo de spaCy (si aún no se instala desde requirements.txt)
RUN python -m spacy download es_core_news_md

# Expón el puerto (ajusta si usas otro)
EXPOSE 8080

# Comando de ejecución
CMD ["python", "app.py"]
#CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
