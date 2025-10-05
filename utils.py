
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
import re

def reduce_dims(df):
    # Reducir dimensiones a 3 con PCA
    pca = PCA(n_components=3)
    emb_cols = [col for col in df.columns if col.startswith("emb_")]
    df_emb = df[emb_cols]
    scaler = MinMaxScaler(feature_range=(-20, 20))
    emb_pca = pca.fit_transform(df_emb)
    emb_scaled = scaler.fit_transform(emb_pca)
    df_pca = pd.DataFrame(emb_scaled, columns=['PC1', 'PC2', 'PC3'])

    df_pca['kepid'] = df['kepid']
    df_pca['kepler_name'] = df['kepler_name']
    df_pca['koi_disposition'] = df['koi_disposition']
    df_pca = df_pca.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df_pca
def find_nearest(candidates, df_full, model, expected_columns, quantity=2):
    emb_cols = [col for col in df_full.columns if col.startswith("emb_")]

    df_full_pca = reduce_dims(df_full)
    df_candidates_pca = reduce_dims(candidates)

    results = []
    
    for pos, (idx, cand) in enumerate(candidates.iterrows()):
        cand_vec = cand[emb_cols].values.reshape(1, -1)
        full_vecs = df_full[emb_cols].values

        sims = cosine_similarity(cand_vec, full_vecs)[0]

        nearest_idx = np.argsort(sims)[::-1]
        nearest = []

        for j in nearest_idx:
            if df_full.iloc[j]['kepid'] == cand['kepid'] or df_full.iloc[j]['koi_disposition'] == 'CANDIDATE':
                continue

            neighbor = df_full_pca.iloc[j].to_dict()
            neighbor["kepid"] = int(df_full.iloc[j]["kepid"])
            neighbor["kepler_name"] = str(df_full.iloc[j]["kepler_name"])
            neighbor["koi_disposition"] = str(df_full.iloc[j]["koi_disposition"])
            neighbor["PC1"] = float(neighbor["PC1"])
            neighbor["PC2"] = float(neighbor["PC2"])
            neighbor["PC3"] = float(neighbor["PC3"])
            neighbor["distance"] = float(1 - sims[j]) * 1000

            nearest.append(neighbor)
            if len(nearest) >= quantity:
                break
        
        # candidate: unir PCA + originales
        candidate_data = df_candidates_pca.iloc[pos].to_dict()
        candidate_data["kepid"] = int(cand["kepid"])
        candidate_data["kepler_name"] = str(cand["kepler_name"])
        candidate_data["koi_disposition"] = str(cand["koi_disposition"])
        candidate_data["PC1"] = float(candidate_data["PC1"])
        candidate_data["PC2"] = float(candidate_data["PC2"])
        candidate_data["PC3"] = float(candidate_data["PC3"])
        candidate_data["id"] = pos

        # ---- PREDECIR ----
        df_pred = pd.DataFrame([candidates.iloc[pos]])
        cols_to_drop = ['koi_pdisposition', 'kepid', 'kepler_name', 'koi_disposition']
        df_pred = df_pred.drop(columns=cols_to_drop, errors="ignore")
        df_pred = df_pred[expected_columns]

        pred = model.predict(df_pred)[0]
        prob = model.predict_proba(df_pred)[0, 1]

        # Añadir predicción al candidato
        candidate_data["prediction"] = int(pred)
        #candidate_data["label"] = "CONFIRMED" if pred == 1 else "FALSE POSITIVE"
        candidate_data["probability"] = float(prob)

        results.append({
            "candidate": candidate_data,
            "nearest": nearest
        })

    return results


def predict(model, df, expected_columns):
    # Copiar DF para no modificar el original
    df_pred = df.copy()

    # Eliminar columnas que no se usan para predecir
    cols_to_drop = [
        'koi_pdisposition', 'kepid','kepler_name'
    ]
    df_pred = df_pred.drop(columns=cols_to_drop, errors="ignore")

    # Reordenar columnas igual que en el entrenamiento
    df_pred = df_pred[expected_columns]

    # Hacer predicciones
    preds = model.predict(df_pred)
    probs = model.predict_proba(df_pred)[:, 1]

    # Crear DF de salida (con columnas originales + resultados)
    df_out = df.copy()
    df_out["prediction"] = preds
    df_out["probability"] = probs

    return df_out

def clean_to_predict(df):

    df = df.drop(columns=[
        "rowid","kepoi_name","kepler_name","koi_vet_date",'koi_vet_stat','koi_disp_prov',
        "koi_comment",
        "koi_ingress","koi_longp",  # no aporta nada porque trae puro NaN
        "koi_fittype",
        'koi_limbdark_mod',  # Esta columna tiene un solo dato por tanto se elimina
        'koi_tce_delivname',  # Pendiente para ver por que NAN
        'koi_trans_mod','koi_model_dof','koi_model_chisq','koi_sage',  # NANs
        'koi_datalink_dvr', 'koi_datalink_dvs',  # DE REPORTES
        'koi_eccen',  # valores 0 o NAN
    ], axis=1)

    # Aplicar funciones de puntuación y limpieza
    df["koi_parm_prov"] = df["koi_parm_prov"].apply(score_parm_prov)

    df['koi_quarters'] = df['koi_quarters'].fillna('0')
    df["koi_quarters"] = df["koi_quarters"].apply(weighted_consecutive)

    df["koi_sparprov"] = df["koi_sparprov"].apply(score_sparprov)

    df['koi_score'] = df.apply(
        lambda row: 0 if (row['koi_disposition'] in ['FALSE POSITIVE', 'CANDIDATE']) 
                            and pd.isna(row['koi_score'])
                    else 0.99 if (row['koi_disposition'] == 'CONFIRMED') 
                            and pd.isna(row['koi_score'])
                    else row['koi_score'],
        axis=1
    )

    # Imputar valores faltantes
    df['impact_missing'] = df['koi_impact'].isna().astype(int)
    df['koi_impact'] = df.groupby('koi_disposition')['koi_impact'].transform(lambda x: x.fillna(x.median()))

    return df

def angular_separation_arcsec(ra1_deg, dec1_deg, ra2_deg, dec2_deg) -> float:
    """Separación angular aproximada en arcsec (para pequeñas separaciones)."""
    # Aproximación euclídea en la esfera: válido para separaciones pequeñas.
    dra = (ra2_deg - ra1_deg) * math.cos(math.radians((dec1_deg + dec2_deg) / 2.0))
    ddec = (dec2_deg - dec1_deg)
    sep_deg = math.hypot(dra, ddec)
    return sep_deg * 3600.0

def folded_epoch_diff_hours(T0i_days, Pi_days, T0j_days, Pj_days) -> float:
    """
    Diferencia de época 'efectiva' basada en fase de j en T0i:
    phase = ((T0i - T0j) / Pj) mod 1 -> distancia a entero -> * Pj (en días), convertimos a horas.
    """
    if np.isnan(T0i_days) or np.isnan(T0j_days) or np.isnan(Pi_days) or np.isnan(Pj_days):
        return np.inf
    if Pj_days <= 0:
        return np.inf
    phase = ((T0i_days - T0j_days) / Pj_days) % 1.0
    phase = min(phase, 1.0 - phase)
    delta_days = phase * Pj_days
    return float(delta_days * 24.0)

def masks_for_split(split_kepids: set):
    # estrellas del split
    star_mask = np.array([int(k) in split_kepids for k in stars_df["kepid"].astype(int).values], dtype=bool)
    # planetas del split: incluye todos los KOIs (labeled y CANDIDATE) de esos kepid
    planet_mask = np.array([int(k) in split_kepids for k in planets_df["kepid"].astype(int).values], dtype=bool)
    # y dentro de ellos, quiénes están etiquetados
    planet_labeled_mask = planet_mask & labeled_mask
    return star_mask, planet_mask, planet_labeled_mask

# función para categorizar la columna koi_parm_prov
#dr25 → catálogo final, el más confiable.  Significa que los parámetros vienen del Data Release 25 (DR25). 
#dr24 → penúltimo catálogo (hubo correcciones posteriores).
#q1_q16 → catálogo intermedio, con datos hasta quarter 16.
def score_parm_prov(value: str) -> int:
    """
    Asigna un puntaje a koi_parm_prov en función del data release y número de quarters.
    """
    score = 0

    # Peso por Data Release
    if "dr25" in value:
        score += 3
    elif "dr24" in value:
        score += 2
    else:
        score += 1  # sin dr (ej: q1_q16_koi)

    # Buscar cuántos quarters usa (ejemplo: q1_q17 -> 17)
    match = re.search(r"q1_q(\d+)", value)
    if match:
        quarters = int(match.group(1))
        score *= quarters  # multiplicamos por los quarters
    return score

def score_tce_deliv(value: str) -> int:
    """
    Asigna un puntaje a koi_tce_delivname en función del Data Release y número de quarters.
    """
    if pd.isna(value):
        return 0  # Si no hay TCE asociado, puntaje 0

    score = 0

    # Peso por Data Release
    if "dr25" in value:
        score += 3
    elif "dr24" in value:
        score += 2
    else:
        score += 1  # sin DR (ej: q1_q16_tce)

    # Buscar cuántos quarters usa (ejemplo: q1_q17 -> 17)
    match = re.search(r"q1_q(\d+)", value)
    if match:
        quarters = int(match.group(1))
        score *= quarters  # multiplicamos por los quarters

    return score

#Función para maximizar las observaciones consecutivas
def weighted_consecutive(value: str) -> int:
    max_count = 0   # máximo de 1s consecutivos
    current_count = 0
    
    for x in value:
        if x == '1':
            current_count += 1
            # opcional: suma de los cuadrados para ponderar más los consecutivos
        else:
            if current_count > max_count:
                max_count = current_count
            current_count = 0
    
    # por si la cadena termina en 1s consecutivos
    max_count = max(max_count, current_count)
    
    return max_count

def score_sparprov(value: str) -> int:
    """
    Asigna un puntaje a koi_sparprov en función de la fuente y completitud de los trimestres.
    Valores más confiables reciben mayor puntaje.
    """
    import re

    if not isinstance(value, str):
        return 0  # si es nan o no es string

    score = 0

    # Ponderación según la fuente
    if "dr25" in value:
        score += 5  # más confiable
    elif "dr24" in value:
        score += 4
    elif "stellar" in value:
        score += 3  # general, pero basada en datos estelares
    elif "Solar" in value:
        score += 1  # estimación muy aproximada
    else:
        score += 2  # otros casos

    # Buscar cuántos quarters usa (ejemplo: q1_q17 -> 17)
    match = re.search(r"q1_q(\d+)", value)
    if match:
        quarters = int(match.group(1))
        score *= quarters  # multiplicamos por el número de quarters

    return score
