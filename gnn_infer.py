# -*- coding: utf-8 -*-
"""
Inferencia GNN heterogénea (Kepler KOIs) para producción.

- Construye subgrafo mínimo alrededor de UNA fila (KOI) de entrada.
- Carga pesos locales del modelo y devuelve el embedding del nodo 'planet'.
- Sin código de entrenamiento.

Autor: Ing. Luis (refactor por M365 Copilot)
"""
import os
import math
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import haversine_distances

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, TransformerConv
from torch_geometric.utils import remove_self_loops

# ---------------------------------------------------------------------
# Utilidades/constantes
# ---------------------------------------------------------------------
ARCSEC_PER_RAD = 206264.806
DEG2RAD = math.pi / 180.0
BKJD_TO_BJD_OFFSET = 2454833.0

# Mantengo la dependencia original:
from utils import folded_epoch_diff_hours  # angular_separation_arcsec no es necesario aquí

DEFAULT_ANGULAR_RADIUS_ARCSEC_NEIGHBOR = 120.0
DEFAULT_EPH_MAX_REL_PERIOD_DIFF = 0.005
DEFAULT_EPH_MAX_EPOCH_DIFF_HOURS = 3.0

DEFAULT_HIDDEN_DIM = 32
DEFAULT_DROPOUT = 0.3

def select_device(prefer: Optional[str] = None) -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ---------------------------------------------------------------------
# Modelo (idéntico a tu entrenamiento, sin cabeza de training)
# ---------------------------------------------------------------------
class HeteroSAGETransformer(nn.Module):
    """
    - Proyección por tipo de nodo -> hidden_dim
    - HeteroConv:
        SAGEConv: hosts, hosted_by, in_system
        TransformerConv (con edge_attr): nearby (star-star), ephemeris (planet-planet)
    - Dos capas de mensaje + BN/Dropout
    - Cabeza 'planet' -> logits (2), pero para embeddings tomamos la activación h2["planet"]
    """
    def __init__(self, in_dims: Dict[str, int], hidden_dim=64, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Proyección inicial por tipo de nodo
        self.lin_in = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim) for ntype, in_dim in in_dims.items()
        })

        # BatchNorm por tipo de nodo
        self.bn1 = nn.ModuleDict({ntype: nn.BatchNorm1d(hidden_dim) for ntype in in_dims})
        self.bn2 = nn.ModuleDict({ntype: nn.BatchNorm1d(hidden_dim) for ntype in in_dims})

        # Capa 1
        self.conv1 = HeteroConv({
            ("star","hosts","planet"): SAGEConv((-1, -1), hidden_dim),
            ("planet","hosted_by","star"): SAGEConv((-1, -1), hidden_dim),
            ("planet","in_system","planet"): SAGEConv((-1, -1), hidden_dim),
            ("star","nearby","star"): TransformerConv((-1, -1), hidden_dim, heads=1, edge_dim=1, beta=True),
            ("planet","ephemeris","planet"): TransformerConv((-1, -1), hidden_dim, heads=1, edge_dim=3, beta=True),
        }, aggr="sum")

        # Capa 2
        self.conv2 = HeteroConv({
            ("star","hosts","planet"): SAGEConv((-1, -1), hidden_dim),
            ("planet","hosted_by","star"): SAGEConv((-1, -1), hidden_dim),
            ("planet","in_system","planet"): SAGEConv((-1, -1), hidden_dim),
            ("star","nearby","star"): TransformerConv((-1, -1), hidden_dim, heads=1, edge_dim=1, beta=True),
            ("planet","ephemeris","planet"): TransformerConv((-1, -1), hidden_dim, heads=1, edge_dim=3, beta=True),
        }, aggr="sum")

        self.head_planet = nn.Linear(hidden_dim, 2)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, return_embeddings=False):
        # Proyección inicial
        #h = {ntype: F.relu(self.lin_inntype) for ntype, x in x_dict.items()}
        h = {ntype: F.relu(self.lin_in[ntype](x)) for ntype, x in x_dict.items()}

        # Capa 1
        h1 = self.conv1(
            h, edge_index_dict,
            edge_attr_dict={
                ("star","nearby","star"): edge_attr_dict.get(("star","nearby","star")),
                ("planet","ephemeris","planet"): edge_attr_dict.get(("planet","ephemeris","planet")),
            } if edge_attr_dict is not None else None
        )
        for ntype in h1:
            h1[ntype] = self.bn1[ntype](h1[ntype])
            h1[ntype] = F.relu(h1[ntype])
            h1[ntype] = F.dropout(h1[ntype], p=self.dropout, training=self.training)

        # Capa 2
        h2 = self.conv2(
            h1, edge_index_dict,
            edge_attr_dict={
                ("star","nearby","star"): edge_attr_dict.get(("star","nearby","star")),
                ("planet","ephemeris","planet"): edge_attr_dict.get(("planet","ephemeris","planet")),
            } if edge_attr_dict is not None else None
        )
        for ntype in h2:
            h2[ntype] = self.bn2[ntype](h2[ntype])
            h2[ntype] = F.relu(h2[ntype])
            h2[ntype] = F.dropout(h2[ntype], p=self.dropout, training=self.training)
        # Logits en planet
        logits_planet = self.head_planet(h2["planet"])

        if return_embeddings:
            return logits_planet, h2  # h2 contiene embeddings por tipo; en particular h2["planet"]
        return logits_planet

# ---------------------------------------------------------------------
# Clase de inferencia/producción
# ---------------------------------------------------------------------
class KOIGNNEmbedder:
    """
    Uso:
        embedder = KOIGNNEmbedder(
            csv_path="./data/con_nulos.csv",
            weights_path="./outputs_gnn/model_state.pt",
            preproc_dir="./outputs_gnn/preproc",   # opcional: imp_* y scl_*; si no existen, se ajustan con el CSV
            device=None,
            hidden_dim=32,
            dropout=0.3,
            ang_radius_arcsec=120.0,
            eph_max_rel_period_diff=0.005,
            eph_max_epoch_diff_hours=3.0
        )
        emb = embedder.embed_row(df.iloc[123])  # np.ndarray shape (hidden_dim,)
    """
    def __init__(
        self,
        csv_path: str,
        weights_path: str,
        preproc_dir: Optional[str] = None,
        device: Optional[str] = None,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
        ang_radius_arcsec: float = DEFAULT_ANGULAR_RADIUS_ARCSEC_NEIGHBOR,
        eph_max_rel_period_diff: float = DEFAULT_EPH_MAX_REL_PERIOD_DIFF,
        eph_max_epoch_diff_hours: float = DEFAULT_EPH_MAX_EPOCH_DIFF_HOURS,
    ):
        self.csv_path = csv_path
        self.weights_path = weights_path
        self.preproc_dir = preproc_dir
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.ang_radius_arcsec = ang_radius_arcsec
        self.eph_max_rel_period_diff = eph_max_rel_period_diff
        self.eph_max_epoch_diff_hours = eph_max_epoch_diff_hours

        self.device = select_device(device)
        print(f"[KOIGNNEmbedder] Dispositivo: {self.device}")

        # 1) Cargar CSV y definir columnas
        self._load_base_data()
        # 2) Cargar/ajustar preprocesadores y transformar features base
        self._setup_preprocessors()
        self._transform_base_features()
        # 3) Construir índice espacial de estrellas
        self._build_spatial_index()
        # 4) Instanciar modelo y cargar pesos
        self._build_model_and_load_weights()

    # ------------------ Datos y features ------------------
    def _load_base_data(self):
        df = pd.read_csv(self.csv_path)
        # Limpieza mínima como en tu script:
        cols_drop = [c for c in ['koi_pdisposition','koi_score','koi_fpflag_ss','koi_fpflag_co'] if c in df.columns]
        if cols_drop:
            df = df.drop(cols_drop, axis=1)

        assert "kepid" in df.columns, "Se requiere la columna 'kepid'."
        assert "koi_disposition" in df.columns, "Se requiere la columna 'koi_disposition'."
        assert "ra" in df.columns and "dec" in df.columns, "Se requieren 'ra' y 'dec'."

        self.df = df

        # Columnas de interés (condicionadas a presencia)
        self.star_feat_cols = [c for c in [
            "ra","dec","koi_steff","koi_slogg","koi_smet","koi_srad","koi_smass","koi_sage",
            "koi_kepmag","koi_gmag","koi_rmag","koi_imag","koi_zmag","koi_jmag","koi_hmag","koi_kmag"
        ] if c in df.columns]

        self.planet_feat_cols = [c for c in [
            "koi_period","koi_time0","koi_time0bk","koi_eccen","koi_longp","koi_impact",
            "koi_duration","koi_ingress","koi_depth","koi_ror","koi_srho","koi_prad",
            "koi_sma","koi_incl","koi_teq","koi_insol","koi_dor",
            "koi_max_sngle_ev","koi_max_mult_ev","koi_model_snr",
            "koi_count","koi_num_transits","koi_bin_oedp_sig","koi_model_dof","koi_model_chisq"
        ] if c in df.columns]

        # Estrellas únicas por kepid
        self.stars_df = self.df.drop_duplicates("kepid").reset_index(drop=True)
        self.stars_df = self.stars_df[["kepid"] + self.star_feat_cols]
        # Planetas: todas las filas
        self.planets_df = self.df.copy()

        # Mapas y auxiliares
        self.kepid2star = {int(k): i for i, k in enumerate(self.stars_df["kepid"].astype(int).values)}
        self.host_star_index = self.planets_df["kepid"].astype(int).map(self.kepid2star).values

        # T0 unificado BJD para ephemeris
        if "koi_time0" in self.planets_df.columns:
            T0_bjd = self.planets_df["koi_time0"].astype(float)
        else:
            T0_bjd = pd.Series([np.nan]*len(self.planets_df))

        if "koi_time0bk" in self.planets_df.columns:
            T0_bkjd = self.planets_df["koi_time0bk"].astype(float)
            T0_bjd = T0_bjd.fillna(T0_bkjd + BKJD_TO_BJD_OFFSET)

        self.planets_df["T0_BJD"] = T0_bjd

        self.P_days = self.planets_df["koi_period"].astype(float).values if "koi_period" in self.planets_df.columns else np.full(len(self.planets_df), np.nan)
        self.T0_days = self.planets_df["T0_BJD"].astype(float).values

    def _setup_preprocessors(self):
        # Intenta cargar los preprocesadores si existen
        self.imp_star = SimpleImputer(strategy="median")
        self.scl_star = StandardScaler()
        self.imp_planet = SimpleImputer(strategy="median")
        self.scl_planet = StandardScaler()

        loaded = False
        if self.preproc_dir and os.path.isdir(self.preproc_dir):
            try:
                import joblib
                self.imp_star = joblib.load(os.path.join(self.preproc_dir, "imp_star.pkl"))
                self.scl_star = joblib.load(os.path.join(self.preproc_dir, "scl_star.pkl"))
                self.imp_planet = joblib.load(os.path.join(self.preproc_dir, "imp_planet.pkl"))
                self.scl_planet = joblib.load(os.path.join(self.preproc_dir, "scl_planet.pkl"))
                loaded = True
                print("[KOIGNNEmbedder] Preprocesadores cargados desde pkl.")
            except Exception as e:
                print(f"[KOIGNNEmbedder] Aviso: no se pudieron cargar preprocesadores: {e}. Se ajustarán con el CSV base.")
        if not loaded:
            # Ajustar en el CSV cargado (útil si infieres sobre el mismo universo)
            X_star_raw = self.stars_df[self.star_feat_cols].astype(float)
            X_planet_raw = self.planets_df[self.planet_feat_cols].astype(float)
            self.imp_star.fit(X_star_raw)
            self.imp_planet.fit(X_planet_raw)
            X_star_imp = self.imp_star.transform(X_star_raw)
            X_planet_imp = self.imp_planet.transform(X_planet_raw)
            self.scl_star.fit(X_star_imp)
            self.scl_planet.fit(X_planet_imp)
            if self.preproc_dir:
                try:
                    os.makedirs(self.preproc_dir, exist_ok=True)
                    import joblib
                    joblib.dump(self.imp_star, os.path.join(self.preproc_dir, "imp_star.pkl"))
                    joblib.dump(self.scl_star, os.path.join(self.preproc_dir, "scl_star.pkl"))
                    joblib.dump(self.imp_planet, os.path.join(self.preproc_dir, "imp_planet.pkl"))
                    joblib.dump(self.scl_planet, os.path.join(self.preproc_dir, "scl_planet.pkl"))
                    print("[KOIGNNEmbedder] Preprocesadores ajustados y guardados.")
                except Exception as e:
                    print(f"[KOIGNNEmbedder] Aviso: no se pudieron guardar preprocesadores: {e}")

    def _transform_base_features(self):
        # Mantiene arrays transformados para acceso rápido
        self.X_star_all = self.scl_star.transform(self.imp_star.transform(self.stars_df[self.star_feat_cols].astype(float)))
        self.X_planet_all = self.scl_planet.transform(self.imp_planet.transform(self.planets_df[self.planet_feat_cols].astype(float)))

    def _build_spatial_index(self):
        # Índice espacial sobre estrellas
        stars_ra_deg = self.stars_df["ra"].astype(float).values
        stars_dec_deg = self.stars_df["dec"].astype(float).values
        self.coords_rad = np.column_stack((np.deg2rad(stars_dec_deg), np.deg2rad(stars_ra_deg)))
        self.tree = BallTree(self.coords_rad, metric='haversine')
        self.radius_rad = (self.ang_radius_arcsec / ARCSEC_PER_RAD)

        # Precalcula vecinos por estrella
        neighbors = self.tree.query_radius(self.coords_rad, r=self.radius_rad, return_distance=False)
        self.star_neighbors = {i: set(int(j) for j in neigh if int(j) != i) for i, neigh in enumerate(neighbors)}

    # ------------------ Modelo ------------------
    def _build_model_and_load_weights(self):
        in_dims = {
            "star": self.X_star_all.shape[1],
            "planet": self.X_planet_all.shape[1],
        }
        self.model = HeteroSAGETransformer(in_dims, hidden_dim=self.hidden_dim, dropout=self.dropout).to(self.device)

        assert os.path.isfile(self.weights_path), f"No existe el archivo de pesos: {self.weights_path}"
        state = torch.load(self.weights_path, map_location=self.device)
        # Permitir que 'state' sea state_dict o {'state_dict': ...}
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        print("[KOIGNNEmbedder] Pesos del modelo cargados.")

    # ------------------ Construcción de subgrafo por fila ------------------
    def _prep_row(self, row_like: Union[pd.Series, dict]) -> pd.Series:
        if isinstance(row_like, dict):
            row = pd.Series(row_like)
        elif isinstance(row_like, pd.Series):
            row = row_like
        else:
            raise TypeError("row_like debe ser pd.Series o dict.")
        missing = [c for c in ["kepid","ra","dec"] if c not in row.index]
        if missing:
            raise ValueError(f"La fila de entrada debe contener {missing}")
        return row

    def _neighbors_of_star(self, star_idx: int, extra_coord: Optional[np.ndarray]=None) -> List[int]:
        """
        Si extra_coord es None -> usa vecinos precalculados del star_idx existente.
        Si extra_coord (dec_rad, ra_rad) pertenece a una estrella NUEVA -> consulta vecinos en BallTree.
        """
        if extra_coord is None:
            return sorted(self.star_neighbors.get(star_idx, []))
        # estrella nueva: buscar vecinos dentro del radio
        idxs = self.tree.query_radius(extra_coord.reshape(1,2), r=self.radius_rad, return_distance=False)[0]
        return sorted(int(i) for i in idxs)

    def _feature_star_from_row(self, row: pd.Series) -> np.ndarray:
        # Construye vector de features de estrella para una estrella nueva (o recomputa para existente)
        vals = row.reindex(self.star_feat_cols)
        X_imp = self.imp_star.transform([vals.astype(float).values])
        X_scl = self.scl_star.transform(X_imp)
        return X_scl[0]

    def _feature_planet_from_row(self, row: pd.Series) -> np.ndarray:
        vals = row.reindex(self.planet_feat_cols)
        X_imp = self.imp_planet.transform([vals.astype(float).values])
        X_scl = self.scl_planet.transform(X_imp)
        return X_scl[0]

    def _build_subgraph_for_row(self, row: pd.Series) -> Tuple[HeteroData, Dict[str, Dict[str, int]]]:
        """
        Devuelve:
            data_sub (HeteroData)
            maps: {
                'star': {'old2new': dict, 'new2old': dict},
                'planet': {'old2new': dict, 'new2old': dict},
                'target_planet_new_idx': int
            }
        """
        # --- Identificador de host ---
        kepid = int(row["kepid"])
        row_ra = float(row["ra"])
        row_dec = float(row["dec"])

        # ¿Estrella existente?
        is_existing_star = kepid in self.kepid2star
        if is_existing_star:
            host_star_old_idx = self.kepid2star[kepid]
            neighbors = self._neighbors_of_star(host_star_old_idx)
            star_old_set = set([host_star_old_idx] + neighbors)
            # features de estrellas: tomamos de X_star_all
            star_feats_list = []
            star_old2new = {}
            for j, sidx in enumerate(sorted(star_old_set)):
                star_old2new[sidx] = j
                star_feats_list.append(self.X_star_all[sidx])
            star_x = np.vstack(star_feats_list)
            # coords para star-star attrs
            coords_subset = self.coords_rad[sorted(star_old_set)]
            # planets incluidos: todos los de estas estrellas
            planets_old_indices = np.where([int(x) in star_old_set for x in self.host_star_index])[0]
            planet_old2new = {int(p): i for i, p in enumerate(planets_old_indices)}
            planet_x = self.X_planet_all[planets_old_indices]

            # ¿El planeta objetivo existe en el dataframe base?
            # Intento localizar índice global (por igualdad de todos los campos no es robusto; probamos por posición si se pasó desde df)
            target_planet_old_idx = None
            if hasattr(row, "name") and isinstance(row.name, (int, np.integer)):
                # nombre de índice del df original (si viene de planets_df)
                if 0 <= row.name < len(self.planets_df):
                    target_planet_old_idx = int(row.name)
            # Si no lo encontramos por index, tratamos de match por kepid + periodo (heurística)
            if target_planet_old_idx is None and "koi_period" in row.index:
                candid = self.planets_df.index[(self.planets_df["kepid"].astype(int)==kepid) &
                                               (np.isclose(self.planets_df["koi_period"].astype(float).values,
                                                           float(row["koi_period"]), rtol=1e-6, atol=1e-6))]
                if len(candid) > 0:
                    target_planet_old_idx = int(candid[0])

            # Si aún no existe (p. ej. fila nueva), lo agregamos como nodo adicional
            extra_planet = None
            if (target_planet_old_idx is None) or (target_planet_old_idx not in planet_old2new):
                extra_planet = self._feature_planet_from_row(row)
                # agregar al final
                planet_x = np.vstack([planet_x, extra_planet])
                # map: asignamos id nuevo al final. Lo marcamos con old_idx = -1 (sentinela)
                planet_old2new[-1] = planet_x.shape[0]-1
                target_planet_new_idx = planet_x.shape[0]-1
            else:
                target_planet_new_idx = planet_old2new[target_planet_old_idx]
        else:
            # Estrella nueva: generamos su feature y buscamos vecinos en el catálogo base
            host_star_feat = self._feature_star_from_row(row)
            extra_coord = np.array([np.deg2rad(row_dec), np.deg2rad(row_ra)])
            neighbors = self._neighbors_of_star(star_idx=-1, extra_coord=extra_coord)
            # Estrellas incluidas: la nueva + vecinos
            star_feats_list = [host_star_feat] + [self.X_star_all[n] for n in neighbors]
            star_x = np.vstack(star_feats_list)
            star_old2new = {-1:0} | {int(n):(i+1) for i, n in enumerate(neighbors)}
            coords_subset = np.vstack([extra_coord.reshape(1,2), self.coords_rad[neighbors]])

            # Planetas incluidos: todos los de estrellas vecinas (la estrella nueva solo tiene el planeta de la fila)
            planets_old_indices = np.where([int(x) in set(neighbors) for x in self.host_star_index])[0]
            planet_x = self.X_planet_all[planets_old_indices]
            planet_old2new = {int(p): i for i, p in enumerate(planets_old_indices)}

            # añadir el planeta de la fila como último
            extra_planet = self._feature_planet_from_row(row)
            planet_x = np.vstack([planet_x, extra_planet])
            planet_old2new[-1] = planet_x.shape[0]-1
            target_planet_new_idx = planet_x.shape[0]-1

        # --- Construcción HeteroData ---
        data = HeteroData()
        data["star"].x = torch.tensor(star_x, dtype=torch.float32)
        data["planet"].x = torch.tensor(planet_x, dtype=torch.float32)

        # Mapas inversos (new->old) útiles para debug
        star_new2old = {v:k for k,v in star_old2new.items()}
        planet_new2old = {v:k for k,v in planet_old2new.items()}

        # --- Aristas star->planet y planet->star ---
        # Necesitamos saber para cada planeta su star_index dentro del SUBGRAFO.
        # Para planetas provenientes del catálogo: usamos host_star_index -> luego star_old2new.
        sp_src, sp_dst = [], []  # star -> planet
        ps_src, ps_dst = [], []  # planet -> star

        # planetas existentes en catálogo:
        for p_old, p_new in planet_old2new.items():
            if p_old == -1:
                # planeta extra (de la fila), host_star es la del row (old_idx puede ser -1 si estrella nueva)
                if is_existing_star:
                    s_old = host_star_old_idx
                else:
                    s_old = -1
            else:
                s_old = int(self.host_star_index[p_old])

            if s_old not in star_old2new:
                # si el host no está en el subgrafo (no debería ocurrir en nuestra construcción), lo omitimos
                continue
            s_new = star_old2new[s_old]
            sp_src.append(s_new); sp_dst.append(p_new)
            ps_src.append(p_new); ps_dst.append(s_new)

        edge_sp = torch.tensor(np.vstack([sp_src, sp_dst]), dtype=torch.long) if len(sp_src)>0 else torch.empty((2,0), dtype=torch.long)
        edge_ps = torch.tensor(np.vstack([ps_src, ps_dst]), dtype=torch.long) if len(ps_src)>0 else torch.empty((2,0), dtype=torch.long)
        data["star","hosts","planet"].edge_index = edge_sp
        data["planet","hosted_by","star"].edge_index = edge_ps

        # --- Aristas planet<->planet in_system (intra-kepid) ---
        # formamos grupos por star (en términos del subgrafo)
        star_to_planets_new = {}
        for p_new, p_old in planet_new2old.items():
            # obtener s_new
            if p_old == -1:
                s_old = host_star_old_idx if is_existing_star else -1
            else:
                s_old = int(self.host_star_index[p_old])
            if s_old not in star_old2new:
                continue
            s_new = star_old2new[s_old]
            star_to_planets_new.setdefault(s_new, []).append(p_new)

        pp_src, pp_dst = [], []
        for _, p_list in star_to_planets_new.items():
            if len(p_list) > 1:
                arr = np.array(p_list, dtype=int)
                a = np.repeat(arr, len(arr)-1)
                b = np.concatenate([np.delete(arr, i) for i in range(len(arr))])
                pp_src.extend(a.tolist()); pp_dst.extend(b.tolist())

        edge_pp_intra = torch.tensor(np.vstack([pp_src, pp_dst]), dtype=torch.long) if len(pp_src)>0 else torch.empty((2,0), dtype=torch.long)
        data["planet","in_system","planet"].edge_index = edge_pp_intra

        # --- Aristas star<->star nearby + atributo distancia (arcsec) ---
        ss_src, ss_dst, ss_attr = [], [], []
        n_stars = data["star"].x.size(0)
        if n_stars > 1:
            # calculamos distancia angular para todas las parejas dentro del subgrafo
            for i in range(n_stars):
                for j in range(n_stars):
                    if i == j: continue
                    sep_rad = haversine_distances(coords_subset[i:i+1], coords_subset[j:j+1])[0,0]
                    sep_arcsec = float(sep_rad * ARCSEC_PER_RAD)
                    if sep_arcsec <= self.ang_radius_arcsec:
                        ss_src.append(i); ss_dst.append(j); ss_attr.append([sep_arcsec])
        edge_ss = torch.tensor(np.vstack([ss_src, ss_dst]), dtype=torch.long) if len(ss_src)>0 else torch.empty((2,0), dtype=torch.long)
        edge_ss_attr = torch.tensor(np.array(ss_attr), dtype=torch.float32) if len(ss_attr)>0 else torch.empty((0,1), dtype=torch.float32)
        data["star","nearby","star"].edge_index = edge_ss
        data["star","nearby","star"].edge_attr  = edge_ss_attr

        # --- Aristas planet<->planet ephemeris-like (entre estrellas cercanas) ---
        epi_src, epi_dst, epi_attr = [], [], []
        # Construimos mapping planeta->(periodo, T0, host_star_new)
        # Para planetas extra (p_old==-1), tomamos P y T0 de la fila (si existen)
        P_map, T0_map, Snew_map = {}, {}, {}
        for p_new, p_old in planet_new2old.items():
            if p_old == -1:
                Pi = float(row["koi_period"]) if "koi_period" in row.index and pd.notna(row["koi_period"]) else np.nan
                T0i = float(row["T0_BJD"]) if "T0_BJD" in row.index and pd.notna(row["T0_BJD"]) else (
                    float(row["koi_time0"]) if "koi_time0" in row.index and pd.notna(row["koi_time0"]) else np.nan
                )
                if ("koi_time0bk" in row.index) and (not np.isfinite(T0i)) and pd.notna(row["koi_time0bk"]):
                    T0i = float(row["koi_time0bk"]) + BKJD_TO_BJD_OFFSET
                if is_existing_star:
                    s_old = host_star_old_idx
                else:
                    s_old = -1
                s_new = star_old2new.get(s_old, None)
            else:
                Pi = float(self.P_days[p_old]) if np.isfinite(self.P_days[p_old]) else np.nan
                T0i = float(self.T0_days[p_old]) if np.isfinite(self.T0_days[p_old]) else np.nan
                s_old = int(self.host_star_index[p_old])
                s_new = star_old2new.get(s_old, None)

            P_map[p_new] = Pi
            T0_map[p_new] = T0i
            Snew_map[p_new] = s_new

        # Para cada par de estrellas vecinas en el subgrafo
        # (usamos star->planets ya calculado)
        for s_i, p_list_i in star_to_planets_new.items():
            for s_j, p_list_j in star_to_planets_new.items():
                if s_i == s_j: 
                    continue
                # verifica que estén cerca (ya se asegura por construcción, pero checamos por si acaso)
                # si no hay arista star-star directa, podemos aun aplicar la condición angular
                # (opcional: omitimos para evitar costo extra)
                for i in p_list_i:
                    Pi, T0i = P_map[i], T0_map[i]
                    if not (np.isfinite(Pi) and Pi > 0 and np.isfinite(T0i)):
                        continue
                    for j in p_list_j:
                        Pj, T0j = P_map[j], T0_map[j]
                        if not (np.isfinite(Pj) and Pj > 0 and np.isfinite(T0j)):
                            continue
                        rel = abs(Pi - Pj) / max(min(Pi, Pj), 1e-6)
                        if rel > self.eph_max_rel_period_diff:
                            continue
                        dT_hours = folded_epoch_diff_hours(T0i, Pi, T0j, Pj)
                        if dT_hours > self.eph_max_epoch_diff_hours:
                            continue
                        # distancia angular estrella-estrella como atributo adicional (usar coords_subset)
                        theta_arcsec = 0.0
                        if s_i < coords_subset.shape[0] and s_j < coords_subset.shape[0]:
                            sep_rad = haversine_distances(coords_subset[s_i:s_i+1], coords_subset[s_j:s_j+1])[0,0]
                            theta_arcsec = float(sep_rad * ARCSEC_PER_RAD)
                        attrs = [float(rel), float(dT_hours), float(theta_arcsec)]
                        epi_src.append(i); epi_dst.append(j); epi_attr.append(attrs)
                        epi_src.append(j); epi_dst.append(i); epi_attr.append(attrs)

        edge_epi = torch.tensor(np.vstack([epi_src, epi_dst]), dtype=torch.long) if len(epi_src)>0 else torch.empty((2,0), dtype=torch.long)
        edge_epi_attr = torch.tensor(np.array(epi_attr), dtype=torch.float32) if len(epi_attr)>0 else torch.empty((0,3), dtype=torch.float32)
        data["planet","ephemeris","planet"].edge_index = edge_epi
        data["planet","ephemeris","planet"].edge_attr  = edge_epi_attr

        # Limpieza de self-loops por si acaso
        for rel in data.edge_types:
            ei = data[rel].edge_index
            ea = data[rel].edge_attr if "edge_attr" in data[rel] else None
            ei2, mask = remove_self_loops(ei)
            data[rel].edge_index = ei2
            if ea is not None and mask is not None:
                data[rel].edge_attr = ea[mask]

        # Mover al dispositivo
        data = self._to_device(data)

        maps = {
            "star": {"old2new": star_old2new, "new2old": star_new2old},
            "planet": {"old2new": planet_old2new, "new2old": planet_new2old},
            "target_planet_new_idx": target_planet_new_idx
        }
        return data, maps

    def _to_device(self, data: HeteroData) -> HeteroData:
        data = data.clone()
        data["star"].x = data["star"].x.to(self.device)
        data["planet"].x = data["planet"].x.to(self.device)
        for rel in data.edge_types:
            data[rel].edge_index = data[rel].edge_index.to(self.device)
            if "edge_attr" in data[rel]:
                data[rel].edge_attr = data[rel].edge_attr.to(self.device)
        return data

    # ------------------ API pública ------------------
    @torch.no_grad()
    def embed_row(self, row_like: Union[pd.Series, dict]) -> np.ndarray:
        """
        Recibe una fila (pd.Series o dict) con al menos: kepid, ra, dec (+ features de star/planet).
        Construye subgrafo, hace forward (eval) y retorna el embedding (np.ndarray) del planeta objetivo.
        """
        row = self._prep_row(row_like)

        # Asegura columnas necesarias para features de planeta/ephemeris
        # (si faltan, serán imputadas por SimpleImputer)
        data, maps = self._build_subgraph_for_row(row)

        logits, hdict = self.model(
            {"star": data["star"].x, "planet": data["planet"].x},
            {
                ("star","hosts","planet"): data["star","hosts","planet"].edge_index,
                ("planet","hosted_by","star"): data["planet","hosted_by","star"].edge_index,
                ("planet","in_system","planet"): data["planet","in_system","planet"].edge_index,
                ("star","nearby","star"): data["star","nearby","star"].edge_index,
                ("planet","ephemeris","planet"): data["planet","ephemeris","planet"].edge_index,
            },
            edge_attr_dict={
                ("star","nearby","star"): data["star","nearby","star"].edge_attr if "edge_attr" in data["star","nearby","star"] else None,
                ("planet","ephemeris","planet"): data["planet","ephemeris","planet"].edge_attr if "edge_attr" in data["planet","ephemeris","planet"] else None,
            },
            return_embeddings=True
        )
        target_idx = maps["target_planet_new_idx"]
        emb = hdict["planet"][target_idx].detach().cpu().numpy()
        return emb

    @torch.no_grad()
    def embed_dataframe(self, df_rows: pd.DataFrame) -> np.ndarray:
        """
        Embebe múltiples filas (itera fila a fila). Devuelve matriz (n_rows, hidden_dim).
        Nota: se construye un subgrafo por fila; si quieres acelerar, agrupa por kepid o reutiliza subgrafos.
        """
        embs = []
        for _, row in df_rows.iterrows():
            embs.append(self.embed_row(row))
        return np.stack(embs, axis=0)
