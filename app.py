

from flask import Flask, request, jsonify
import numpy as np
import traceback
from flask_cors import CORS
app = Flask(__name__)
import math
import pandas as pd
import shap
import os
import requests
import io
import pickle
CORS(app, origins=["http://localhost:4200",'https://tfm.grijalvaromero.dev'])
from utils import reduce_dims, find_nearest, predict, clean_to_predict
from db import Database
from datetime import datetime
from gnn_infer import KOIGNNEmbedder

##LOAD DATA
df = pd.read_csv("./data/prod.csv")
candidates = df[df['koi_disposition'] == 'CANDIDATE'].copy()

#CONFIG DATA
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
#app.config["UPLOAD_FOLDER"] = "./uploads"
db = Database()
##LOAD MODEL

embedder = KOIGNNEmbedder(
    csv_path="./data/con_nulos.csv",
    weights_path="./outputs_gnn/model_state.pt",
    preproc_dir="./outputs_gnn/preproc",
    hidden_dim=32,
    dropout=0.3,
    ang_radius_arcsec=120.0,
    eph_max_rel_period_diff=0.005,
    eph_max_epoch_diff_hours=3.0
)
artefacto=[]
try:
    with open('./models/simple.pkl', 'rb') as file:
        artefacto = pickle.load(file)
except Exception as e:
    print("Error al cargar el modelo:")
    traceback.print_exc()
    
model = artefacto['model']
expected_columns = artefacto['columns']
#X_train = artefacto['X_train']

@app.route('/')
def home():
    return "API de predicción de precios de casas"



@app.route('/data')
def data():
    data = reduce_dims(df.head(50))
    return jsonify({
        "data": data.to_dict(orient='records'),
    })

@app.route('/nearest')
def nearest():
    data = find_nearest(candidates.head(30),df,model, expected_columns,quantity=2)
    #predicts = predict(model, candidates.head(30),expected_columns)
    return jsonify({
        #"data":predicts.to_dict(orient='records')
        "data":data
    })

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No se envió archivo"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Nombre de archivo inválido"}), 400

    # Generar nombre con fecha y hora
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"

    # Guardar archivo
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Insertar registro en la tabla uploads
    db.insert_upload(filename)

    return jsonify({"message": f"Archivo {filename} guardado y registrado en BD"}), 200

@app.route("/uploads", methods=["GET"])
def get_uploads():
    conn = db.connect()
    cur = conn.cursor()
    cur.execute("SELECT id, file, created_at, updated_at FROM uploads ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()

    # Convertir filas a lista de diccionarios
    uploads = [
        {"id": r[0], "file": r[1], "created_at": r[2], "updated_at": r[3]}
        for r in rows
    ]

    return jsonify({
        "data":uploads
    }), 200

@app.route("/get_csv/<int:id>", methods=["GET"])
def get_csv(id):
    filename = db.get_filename_by_id(id)
    if not filename:
        return jsonify({"error": f"Archivo no encontrado: {filename}"}), 404

    # Construir la ruta completa
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "Archivo no existe en el servidor"}), 404
    df_2 = pd.read_csv(filepath)
    backup = clean_to_predict(df_2)
  
    #emb = embedder.embed_row(backup.iloc[0]) 

    embeddings = []

    for i, row in backup.iterrows():
        emb = embedder.embed_row(row)  # ndarray o tensor
        emb_list = emb.tolist() if hasattr(emb, 'tolist') else list(emb)

        embeddings.append(emb_list)
    
    emb_df = pd.DataFrame(embeddings)
    # Opcional: renombrar columnas como emb_0, emb_1, ...
    emb_df.columns = [f"emb_{i}" for i in range(emb_df.shape[1])]    
    emb_df["kepid"] = df["kepid"]
    #emb_df["kepoi_name"] = df["kepoi_name"]
    emb_df["kepler_name"] = df["kepler_name"]
    emb_df["koi_disposition"] = df["koi_disposition"]


    #print(emb_df.head())
    
    res = find_nearest(emb_df, df,model, expected_columns,quantity=2)
  
    return jsonify({"data": res}), 200
   

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    #$app.run(host='0.0.0.0', port=8001,debug=True)
    