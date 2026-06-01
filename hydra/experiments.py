"""Experimentos de revisión (programa #1-#6) para el paper de sparse sensing sobre
el dataset UCI Condition Monitoring of Hydraulic Systems.

Pensado para Colab: cacheable (JSON firmado), paralelizable en CPU (joblib) y GPU
(KAN/torch), con salida bonita (rich/tqdm) orquestada desde el notebook.

#1 forward-selection SOLO-FÍSICO (excluye sensores virtuales SE/CE/CP)
#2 multiseed 30 seeds + tamaños de efecto
#3 tuning UNIFORME (mismo presupuesto random-search para todas las familias)
#5 expresión simbólica KAN (2 sensores)
#6 Friedman + Nemenyi (diagrama Critical Difference) + medianas/IQR
"""
from __future__ import annotations
import hashlib
import json
import os
import statistics
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .io import load_raw_dataset, SENSOR_NAMES, PROFILE_COLUMNS
from .features import build_feature_matrix, FEATURE_NAMES

UCI_URL = ("https://archive.ics.uci.edu/static/public/447/"
           "condition+monitoring+of+hydraulic+systems.zip")
VIRTUAL_SENSORS = ["SE", "CE", "CP"]          # eficiencia/refrigeración: señales virtuales
PHYSICAL_SENSORS = [s for s in SENSOR_NAMES if s not in VIRTUAL_SENSORS]
TARGETS = {"pump_leak": 2, "accum": 3}        # índice de columna en profile
CACHE_DIR = os.environ.get("HYDRA_CACHE", "hydra_cache")


# --------------------------------------------------------------------------- #
# datos + features                                                            #
# --------------------------------------------------------------------------- #
def ensure_data(data_dir="data_raw"):
    """Descarga y descomprime el dataset UCI 447 si falta."""
    data_dir = Path(data_dir)
    if (data_dir / "profile.txt").exists():
        return data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    zp = data_dir / "uci447.zip"
    urllib.request.urlretrieve(UCI_URL, zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(data_dir)
    # algunos zips traen un zip interno; descomprimir cualquier .zip resultante
    for inner in data_dir.glob("*.zip"):
        if inner != zp:
            with zipfile.ZipFile(inner) as z:
                z.extractall(data_dir)
    return data_dir


def build_xy(data_dir="data_raw", cache=CACHE_DIR):
    """Devuelve (X, feature_names, targets_dict). Cachea X en .npz."""
    os.makedirs(cache, exist_ok=True)
    xp = Path(cache) / "Xfeatures.npz"
    sensors, profile = load_raw_dataset(ensure_data(data_dir))
    if xp.exists():
        d = np.load(xp, allow_pickle=True)
        X, feature_names = d["X"], list(d["names"])
    else:
        X, feature_names = build_feature_matrix(sensors)
        np.savez_compressed(xp, X=X, names=np.array(feature_names, dtype=object))
    targets = {name: profile.iloc[:, col].to_numpy() for name, col in TARGETS.items()}
    return X, feature_names, targets


def block_columns(feature_names):
    """sensor -> lista de índices de sus 15 columnas."""
    cols = {}
    for j, fn in enumerate(feature_names):
        s = fn.rsplit("_", 1)[0]
        cols.setdefault(s, []).append(j)
    return cols


# --------------------------------------------------------------------------- #
# cache firmado                                                               #
# --------------------------------------------------------------------------- #
def _sig(**kw):
    return hashlib.sha1(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()[:16]


def cached(tag, **kw):
    """Devuelve (path, datos|None). Si existe y la firma coincide, datos!=None."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    sig = _sig(**kw)
    p = Path(CACHE_DIR) / f"{tag}_{sig}.json"
    if p.exists():
        try:
            return p, json.load(open(p, encoding="utf-8"))
        except Exception:
            pass
    return p, None


def save_cache(path, data):
    json.dump(data, open(path, "w", encoding="utf-8"))


# --------------------------------------------------------------------------- #
# modelo rápido para selección/score                                          #
# --------------------------------------------------------------------------- #
def _lgbm(seed=42):
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                              n_jobs=1, random_state=seed, verbosity=-1)


def _cv_f1(X, y, seed=42, folds=4, jobs=1):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return float(cross_val_score(_lgbm(seed), X, y, cv=skf,
                                 scoring="f1_macro", n_jobs=jobs).mean())


# --------------------------------------------------------------------------- #
# #1 forward block selection (con exclusión de virtuales)                     #
# --------------------------------------------------------------------------- #
def forward_block_selection(X, feature_names, y, candidate_sensors,
                            target_f1=0.95, seed=42, max_blocks=8, progress=None, jobs=-1):
    """Greedy: agrega el bloque-sensor que más sube el macro-F1 (CV) hasta llegar
    a target_f1 o max_blocks. Devuelve la curva [(sensor, f1)].

    Paraleliza la evaluacion de los sensores candidatos sobre TODOS los nucleos
    (joblib jobs=-1); cada CV interno corre con 1 nucleo para no sobre-suscribir."""
    from joblib import Parallel, delayed
    cols = block_columns(feature_names)
    chosen, used_cols, curve = [], [], []
    remaining = list(candidate_sensors)
    while remaining and len(chosen) < max_blocks:
        scores = Parallel(n_jobs=jobs)(
            delayed(_cv_f1)(X[:, used_cols + cols[s]], y, seed, 4, 1) for s in remaining)
        bi = int(max(range(len(remaining)), key=lambda i: scores[i]))
        best_s, best_f1 = remaining[bi], scores[bi]
        used_cols = used_cols + cols[best_s]
        chosen.append(best_s); remaining.remove(best_s)
        curve.append((best_s, round(best_f1, 4)))
        if progress:
            progress(1)
        if best_f1 >= target_f1:
            break
    return curve


def run_sparse_study(X, feature_names, targets, seed=42, on_step=None):
    """#1: compara forward-selection con TODOS los sensores vs SOLO-FÍSICOS."""
    out = {}
    for tname, y in targets.items():
        out[tname] = {
            "all": forward_block_selection(X, feature_names, y, sorted(block_columns(feature_names)),
                                           seed=seed, progress=on_step),
            "physical_only": forward_block_selection(X, feature_names, y, PHYSICAL_SENSORS,
                                                     seed=seed, progress=on_step),
        }
    return out


# --------------------------------------------------------------------------- #
# #3 zoo de modelos con grillas para tuning UNIFORME (random-search)          #
# --------------------------------------------------------------------------- #
def model_zoo():
    """family -> (estimator_factory, param_distribution). Mismo presupuesto para todos."""
    from scipy.stats import randint, uniform
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    zoo = {
        "DecisionTree": (lambda: DecisionTreeClassifier(random_state=0),
                         {"max_depth": [3, 5, 10, 20, None], "criterion": ["gini", "entropy"],
                          "min_samples_split": randint(2, 20)}),
        "RandomForest": (lambda: RandomForestClassifier(random_state=0, n_jobs=1),
                         {"n_estimators": randint(100, 300), "max_depth": [5, 10, 20, None],
                          "max_features": ["sqrt", "log2", 0.5]}),
        "ExtraTrees": (lambda: ExtraTreesClassifier(random_state=0, n_jobs=1),
                       {"n_estimators": randint(100, 300), "max_depth": [5, 10, 20, None],
                        "max_features": ["sqrt", "log2", 0.5]}),
        "SVM": (lambda: SVC(random_state=0),
                {"kernel": ["linear", "rbf", "poly"], "C": uniform(0.1, 100), "gamma": ["scale", "auto"]}),
        "KNN": (lambda: KNeighborsClassifier(n_jobs=1),
                {"n_neighbors": randint(3, 20), "weights": ["uniform", "distance"],
                 "metric": ["euclidean", "manhattan"]}),
        "HistGB": (lambda: HistGradientBoostingClassifier(random_state=0),
                   {"max_iter": randint(100, 500), "max_depth": [3, 5, 10, None],
                    "learning_rate": uniform(0.01, 0.3)}),
    }
    try:
        import lightgbm as lgb
        zoo["LightGBM"] = (lambda: lgb.LGBMClassifier(n_jobs=1, random_state=0, verbosity=-1),
                           {"n_estimators": randint(100, 300), "num_leaves": randint(15, 63),
                            "learning_rate": uniform(0.01, 0.3)})
    except Exception:
        pass
    try:
        import xgboost as xgb
        zoo["XGBoost"] = (lambda: xgb.XGBClassifier(n_jobs=1, random_state=0, verbosity=0),
                          {"n_estimators": randint(100, 300), "max_depth": randint(3, 10),
                           "learning_rate": uniform(0.01, 0.3)})
    except Exception:
        pass
    return zoo


def _pipe(estimator_factory):
    """Pipeline con StandardScaler + estimador. Escalar es CRÍTICO para SVM/KNN
    (si no, SVM sobre features sin escalar se vuelve lentísimo / no converge) y es
    inocuo para árboles/boosting. El scaler se ajusta dentro de cada fold (sin fuga)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("sc", StandardScaler()), ("clf", estimator_factory())])


def tune_uniform(estimator_factory, param_dist, X, y, n_iter=80, seed=0):
    """RandomizedSearchCV con presupuesto fijo (mismo para todas las familias),
    sobre un pipeline escalado. Devuelve best_params_ con prefijo 'clf__'."""
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    pdist = {f"clf__{k}": v for k, v in param_dist.items()}
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    rs = RandomizedSearchCV(_pipe(estimator_factory), pdist, n_iter=n_iter, scoring="f1_macro",
                            cv=skf, random_state=seed, n_jobs=-1, refit=True)
    rs.fit(X, y)
    return rs.best_params_, float(rs.best_score_)


# --------------------------------------------------------------------------- #
# #2 multiseed + tamaños de efecto                                            #
# --------------------------------------------------------------------------- #
def multiseed_f1(estimator_factory, params, X, y, n_seeds=30, jobs=-1):
    """Macro-F1 de test sobre n_seeds splits 80/20 estratificados. Paralelo."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from joblib import Parallel, delayed

    def one(seed):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        m = _pipe(estimator_factory).set_params(**params)   # params vienen con prefijo 'clf__'
        m.fit(Xtr, ytr)                                      # el pipeline escala con fit en train
        return float(f1_score(yte, m.predict(Xte), average="macro"))

    return list(Parallel(n_jobs=jobs)(delayed(one)(s) for s in range(42, 42 + n_seeds)))


def cliffs_delta(a, b):
    """Tamaño de efecto Cliff's delta entre dos muestras."""
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


# --------------------------------------------------------------------------- #
# #6 Friedman + Nemenyi (CD) + medianas/IQR                                   #
# --------------------------------------------------------------------------- #
def friedman_nemenyi(perf):
    """perf: dict family -> list de F1 por seed (mismas seeds). Devuelve stats + CD."""
    import numpy as np
    from scipy.stats import friedmanchisquare
    fams = list(perf)
    M = np.array([perf[f] for f in fams])          # families x seeds
    stat, p = friedmanchisquare(*M)
    # ranks promedio (mayor F1 = mejor = rank 1)
    ranks = (-M).argsort(axis=0).argsort(axis=0) + 1
    avg_rank = {f: float(ranks[i].mean()) for i, f in enumerate(fams)}
    k, n = len(fams), M.shape[1]
    q05 = {2:1.960,3:2.343,4:2.569,5:2.728,6:2.850,7:2.949,8:3.031,9:3.102,10:3.164}
    cd = q05.get(k, 3.2) * (k * (k + 1) / (6.0 * n)) ** 0.5
    nem = None
    try:
        import scikit_posthocs as sp
        nem = sp.posthoc_nemenyi_friedman(M.T).values.tolist()
    except Exception:
        pass
    return {"stat": float(stat), "p": float(p), "avg_rank": avg_rank,
            "cd": float(cd), "k": k, "n": n, "families": fams, "nemenyi": nem}


def describe(vals):
    s = sorted(vals)
    q1, q3 = np.percentile(s, [25, 75])
    return {"mean": float(np.mean(s)), "std": float(np.std(s, ddof=1)),
            "median": float(np.median(s)), "iqr": float(q3 - q1),
            "min": float(s[0]), "max": float(s[-1])}


# --------------------------------------------------------------------------- #
# #5 KAN simbólico (2 sensores)                                               #
# --------------------------------------------------------------------------- #
def kan_symbolic(X, feature_names, y, sensors=("SE", "TS2"), seed=42, steps=120):
    """Entrena un KAN compacto sobre los features de 2 sensores y recupera la
    expresión simbólica (auto_symbolic). Usa GPU si hay. Devuelve dict con F1 y fórmulas."""
    import torch
    from kan import KAN
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    cols = block_columns(feature_names)
    idx = [c for s in sensors for c in cols[s]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Xte, ytr, yte = train_test_split(X[:, idx], y, test_size=0.2, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    classes = sorted(set(y.tolist())); cmap = {c: i for i, c in enumerate(classes)}
    yi_tr = np.array([cmap[v] for v in ytr]); yi_te = np.array([cmap[v] for v in yte])
    ds = {"train_input": torch.tensor(Xtr, dtype=torch.float32, device=dev),
          "test_input": torch.tensor(Xte, dtype=torch.float32, device=dev),
          "train_label": torch.tensor(yi_tr, dtype=torch.long, device=dev),
          "test_label": torch.tensor(yi_te, dtype=torch.long, device=dev)}
    model = KAN(width=[len(idx), 4, len(classes)], grid=5, k=3, seed=seed, device=dev)
    def loss_fn(pred, tgt):
        return torch.nn.functional.cross_entropy(pred, tgt)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        out = model(ds["train_input"])
        loss = loss_fn(out, ds["train_label"]); loss.backward(); opt.step()
    with torch.no_grad():
        pred = model(ds["test_input"]).argmax(1).cpu().numpy()
    f1 = float(f1_score(yi_te, pred, average="macro"))
    formulas = None
    try:
        model.auto_symbolic(lib=["x", "x^2", "exp", "sin", "tanh"])
        formulas = [str(model.symbolic_formula()[0][i]) for i in range(len(classes))]
    except Exception as e:
        formulas = [f"(auto_symbolic falló: {e})"]
    return {"sensors": list(sensors), "n_features": len(idx), "test_f1_macro": round(f1, 4),
            "device": dev, "formulas": formulas}
