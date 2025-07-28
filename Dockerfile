# Usa imagen oficial de Python
FROM python:3.11
#pip freeze | sed '/@ file:/d' > requirements.txt
# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos
COPY . /app
COPY ./models /app/models
# Instala git y otras dependencias necesarias
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Instala requerimientos
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache /tmp/*

RUN mkdir -p /app/data /app/models

# Descargar archivos desde tu hosting
RUN wget -O /app/data/prices_avg_mt2.csv https://tfm.grijalvaromero.dev/models/prices_avg_mt2.csv && \
    wget -O /app/models/simple.pkl https://tfm.grijalvaromero.dev/models/simple.pkl


# Expón el puerto (ajusta si usas otro)
EXPOSE 8080

# Comando de ejecución
CMD ["python", "app.py"]
#CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]