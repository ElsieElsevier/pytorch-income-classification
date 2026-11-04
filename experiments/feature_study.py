"""One-factor-at-a-time study on a FIXED stratified validation split.

Usage:
  feature_study.py                 -> all configs, 3 seeds, validation metrics only (test never scored)
  feature_study.py --test a,b,c    -> named configs, 5 seeds, also scores test
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
import copy
import json
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
R = str(HERE.parent) + "/"
CAT = ["workclass", "marital_status", "occupation", "relationship", "race", "gender", "native_country"]
NUM = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
SPLIT_SEED = 42


def load():
    tr = pd.read_csv(R + "training_data.csv", index_col=0)
    te = pd.read_csv(R + "test_data.csv", index_col=0)
    te = te[te["workclass"].notna()].copy()
    te["age"] = te["age"].astype(int)
    for d in (tr, te):
        for c in CAT + ["income_bracket"]:
            d[c] = d[c].astype(str).str.strip()
        d["y"] = d["income_bracket"].str.rstrip(".").eq(">50K").astype(np.float32)
    return tr, te


def prep(tr, te, unknown=False, log=False, flags=False, bins=False):
    """All statistics are learned from the training portion only."""
    a, b = train_test_split(tr, test_size=0.2, stratify=tr.y, random_state=SPLIT_SEED)
    a, b, t = a.copy(), b.copy(), te.copy()
    for c in CAT:
        fill = "Unknown" if unknown else a.loc[a[c] != "?", c].mode()[0]
        for d in (a, b, t):
            d[c] = d[c].replace("?", fill)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(a[CAT])
    # Bin edges use the raw values; zero-inflated capital columns take quantiles of non-zero values
    edges = {}
    if bins:
        for c in NUM:
            v = a.loc[a[c] != 0, c] if c.startswith("capital") else a[c]
            edges[c] = np.unique(np.quantile(v, np.linspace(0, 1, 11)[1:-1]))

    def extra(d):
        out = []
        if flags:
            out += [(d.capital_gain.values > 0).astype(float), (d.capital_loss.values > 0).astype(float)]
        return np.column_stack(out) if out else np.empty((len(d), 0))

    def binned(d):
        if not bins:
            return np.empty((len(d), 0))
        parts = []
        for c in NUM:
            k = np.digitize(d[c].values, edges[c])
            parts.append(np.eye(len(edges[c]) + 1)[k])
        return np.hstack(parts)

    def numeric(d):
        x = d[NUM].astype(float).copy()
        if log:
            x["capital_gain"] = np.log1p(x["capital_gain"])
            x["capital_loss"] = np.log1p(x["capital_loss"])
        return x

    sc = StandardScaler().fit(numeric(a))
    f = lambda d: np.hstack([sc.transform(numeric(d)), enc.transform(d[CAT]), extra(d), binned(d)]).astype(np.float32)
    return f(a), a.y.values, f(b), b.y.values, f(t), t.y.values


def net(n, logits, dropout):
    layers, prev = [], n
    for h, p in zip((128, 64, 32), dropout):
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    if not logits:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def fit(data, seed, lr=1e-3, logits=False, wd=0.0, sched=False, dropout=(.4, .3, .2)):
    Xa, ya, Xb, yb, Xt, yt = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    T = lambda x: torch.tensor(x, dtype=torch.float32)
    dl = DataLoader(TensorDataset(T(Xa), T(ya).view(-1, 1)), batch_size=64, shuffle=True)
    Xb_, yb_ = T(Xb), T(yb).view(-1, 1)
    m = net(Xa.shape[1], logits, dropout)
    crit = nn.BCEWithLogitsLoss() if logits else nn.BCELoss()
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    s = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=3) if sched else None
    best, best_ep, cnt, state = float("inf"), 0, 0, None
    for ep in range(100):
        m.train()
        for x, y in dl:
            loss = crit(m(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = crit(m(Xb_), yb_).item()
        if s:
            s.step(vl)
        if vl < best - 1e-5:
            best, best_ep, cnt = vl, ep + 1, 0
            state = copy.deepcopy(m.state_dict())
        else:
            cnt += 1
            if cnt >= 15:
                break
    m.load_state_dict(state)
    m.eval()

    def predict(X):
        with torch.no_grad():
            o = m(T(X)).view(-1)
            return (torch.sigmoid(o) if logits else o).numpy()

    return predict, best_ep


def score(y, p):
    h = (p > .5).astype(int)
    return dict(logloss=log_loss(y, p), auc=roc_auc_score(y, p), acc=100 * accuracy_score(y, h), f1=f1_score(y, h),
                prec=precision_score(y, h), rec=recall_score(y, h))


LOW = (.1, .1, .1)
CONFIGS = {
    "base": ({}, {}),
    "unknown": (dict(unknown=True), {}),
    "log1p": (dict(log=True), {}),
    "logits": ({}, dict(logits=True)),
    "weight_decay": ({}, dict(wd=1e-4)),
    "scheduler": ({}, dict(sched=True)),
    "D_bundle": (dict(unknown=True, log=True), dict(logits=True, wd=1e-4, sched=True)),
    "low_dropout": ({}, dict(dropout=LOW)),
    "flags": (dict(flags=True), {}),
    "bins": (dict(bins=True), {}),
    "bins+low_dropout": (dict(bins=True), dict(dropout=LOW)),
    "bins+low_dropout+unknown": (dict(bins=True, unknown=True), dict(dropout=LOW)),
    "bins+low_dropout+sched": (dict(bins=True), dict(dropout=LOW, sched=True)),
    "bins+low_dropout+unknown+logits+sched": (dict(bins=True, unknown=True), dict(dropout=LOW, logits=True, sched=True)),
}

tr, te = load()
use_test = len(sys.argv) > 2 and sys.argv[1] == "--test"
names = sys.argv[2].split(",") if use_test else list(CONFIGS)
seeds = range(5) if use_test else range(3)
rows = []

base_data = prep(tr, te)
g = HistGradientBoostingClassifier(random_state=0).fit(base_data[0], base_data[1])
print("hgb val", {k: round(v, 4) for k, v in score(base_data[3], g.predict_proba(base_data[2])[:, 1]).items()}, flush=True)
if use_test:
    print("hgb test", {k: round(v, 4) for k, v in score(base_data[5], g.predict_proba(base_data[4])[:, 1]).items()}, flush=True)

for name in names:
    pk, fk = CONFIGS[name]
    data = prep(tr, te, **pk)
    for s in seeds:
        predict, ep = fit(data, s, **fk)
        split = "test" if use_test else "val"
        y, X = (data[5], data[4]) if use_test else (data[3], data[2])
        r = dict(config=name, seed=s, best_epoch=ep, features=data[0].shape[1], **score(y, predict(X)))
        rows.append(r)
        print(split, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)

d = pd.DataFrame(rows)
summ = d.groupby("config", sort=False).agg(features=("features", "first"), best_ep=("best_epoch", "mean"),
                                           logloss=("logloss", "mean"), auc=("auc", "mean"), auc_sd=("auc", "std"),
                                           acc=("acc", "mean"), acc_sd=("acc", "std"), f1=("f1", "mean"),
                                           prec=("prec", "mean"), rec=("rec", "mean")).round(4)
print("\nSUMMARY", "TEST" if use_test else "VALIDATION")
print(summ.to_string())
json.dump(rows, open(RESULTS / ("feature_study_test.json" if use_test else "feature_study_val.json"), "w"), indent=1)
