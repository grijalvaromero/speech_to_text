# app.py
from flask import Flask, request, jsonify
#import joblib
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
from sklearn.ensemble import GradientBoostingRegressor
CORS(app, origins=["http://localhost:3000",'https://tfm.grijalvaromero.dev'])
# Ruta remota base
REMOTE_BASE_URL = "https://tfm.grijalvaromero.dev/"

# URLs de los archivos
DF_AVG_URL = f"{REMOTE_BASE_URL}models/prices_avg_mt2.csv"
MODEL_URL = f"{REMOTE_BASE_URL}models/simple.pkl"

# Rutas locales
DF_AVG_LOCAL = "./data/prices_avg_mt2.csv"
MODEL_LOCAL = "./models/simple.pkl"

# Crea carpetas si no existen
os.makedirs(os.path.dirname(DF_AVG_LOCAL), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_LOCAL), exist_ok=True)

# ----------------------------
# Cargar CSV (descargar si no existe)
# ----------------------------
if not os.path.exists(DF_AVG_LOCAL):
    print(f"Descargando df_avg desde: {DF_AVG_URL}")
    df_avg_response = requests.get(DF_AVG_URL)
    with open(DF_AVG_LOCAL, 'wb') as f:
        f.write(df_avg_response.content)
    print("Archivo CSV descargado y guardado.")
else:
    print("Archivo CSV ya existe. Cargando localmente.")

#df_avg = pd.read_csv(DF_AVG_LOCAL)

# ----------------------------
# Cargar modelo (descargar si no existe)
# ----------------------------
if not os.path.exists(MODEL_LOCAL):
    print(f"Descargando modelo desde: {MODEL_URL}")
    model_response = requests.get(MODEL_URL)
    with open(MODEL_LOCAL, 'wb') as f:
        f.write(model_response.content)
    print("Modelo guardado localmente.")
else:
    print("Modelo ya existe. Cargando localmente.")

#artefacto = joblib.load(MODEL_LOCAL)
df_avg = pd.read_csv('./data/prices_avg_mt2.csv')
#artefacto = pickle.load('./models/simple.pkl')
artefacto=[]
with open('./models/simple.pkl', 'rb') as file:
    artefacto = pickle.load(file)

model = artefacto['model']
expected_columns = artefacto['columns']
X_train = artefacto['X_train']
@app.route('/')
def home():
    return "API de predicción de precios de casas"

def prepareCols(data):
    from datetime import datetime

    # Accesos rápidos
    form = data["formData"]
    amenities = {a["text"]: a["value"] for a in data["amenitys"]}
    geo = data.get("geoData", {})
    features = data.get("features", {}).get("counts", {})

    # Utilidades
    def safe_div(a, b):
        return a / b if b != 0 else 0

    build_year = amenities.get("Año construcción", datetime.now().year)
    area = amenities.get("Área construida (m²)", 1)
    terrain_area = amenities.get("Área del terreno (m²)", 1)
    price_per_m2 = safe_div(form.get("price", 0), area)
    age = datetime.now().year - build_year
    rooms = amenities.get("Cuartos en total", 0)
    bathrooms = amenities.get("Baños", 0)
    rooms_per_bathroom = safe_div(rooms, bathrooms)

    radius = data.get("features", {}).get("radio", 1)  # en metros
    area_r = math.pi * (radius**2)
    id_distrito =  form.get("distrito", {}).get("id", 0)
    price_temp = df_avg[df_avg['id_distrito'] == id_distrito]['price_per_m2']
    price_mt2 = 0
    if not price_temp.empty:
        price_mt2= price_temp.iloc[0]


    cols = {
        "floors": amenities.get("Niveles", 0),
        "rooms": rooms,
        "bathrooms": bathrooms,
        "importance": geo.get("importance", 0),
        "place_rank": geo.get("place_rank", 0),
        "build_year": build_year,

        "id_distrito": form.get("distrito", {}).get("id", 0),
        "garages": amenities.get("Garajes exteriores", 0) + amenities.get("Garajes Interiores", 0),
        "terrain_area": terrain_area,
        "area": area,
        "price_per_m2": price_per_m2,
        "age": age,
        "rooms_per_bathroom": rooms_per_bathroom,

        "amenity_density": safe_div(features.get("amenity_count", 0), area_r),
        "leisure_density": safe_div(features.get("leisure_count", 0), area_r),
        "shop_density": safe_div(features.get("shop_count", 0), area_r),
        "building_density": safe_div(features.get("building_count", 0), area_r),
        "road_density": safe_div(features.get("road_count", 0), area_r),
        
        "amenity_count": features.get("amenity_count", 0),
        "zip_code": int(form.get("zip_code", geo.get("zip_code", 0))),
        "place_type_num": 1 if geo.get("place_type") == "residential" else 0,
        "lat": form.get("lat", 0),
        "lng": form.get("lng", 0),
        "expense": 0  # puedes ajustar este valor si tienes info
    }

    return cols

@app.route('/predict', methods=['POST'])
def predict():
   
    try:
        data = request.get_json()
        cols = prepareCols(data)
        #return jsonify({
        #   'expected_columns':expected_columns,
        #   'cols':cols   
        #})
        # Asegurarnos de que vienen todas las columnas en el orden correcto
        X_input = [cols.get(col, 0) for col in expected_columns]
        X_array = np.array([X_input])
        X_df = pd.DataFrame(X_array, columns=expected_columns)
        # Predecir en log
        y_log = model.predict(X_array)
        # Volver a escala real
        y_real = np.expm1(y_log)[0]
        RMSE = 236_459
        lower = max(y_real - RMSE, 0)
        upper = y_real + RMSE
        

        # SHAP
        explainer = shap.Explainer(model.predict, X_train)
        shap_values = explainer(X_df)

        # Extraer justificaciones
        explicaciones = []
        for name, value, impact in zip(
            shap_values[0].feature_names,
            shap_values[0].data,
            shap_values[0].values
        ):
            #if abs(impact) < 1000:
                #continue
            explicaciones.append({
                'feature': name,
                'valor': value,
                'impacto': impact,
                'signo': '+' if impact > 0 else '-',
                'razon': f"{'Aumenta' if impact > 0 else 'Reduce'} el valor por {name} = {value}"
            })


        top_explicaciones = sorted(explicaciones, key=lambda x: abs(x['impacto']), reverse=True)[:4]
        return jsonify({
            'predicted_price': float(y_real),
            'currency': 'USD',
            'confidence_range': {
                'lower': round(lower),
                'upper': round(upper)
            },
            "LOG":y_log[0],
            'justificacion': top_explicaciones,
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        }), 400

if __name__ == '__main__':
    #port = int(os.environ.get("PORT", 8080))
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=8080)