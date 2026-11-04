"""Multi-seed comparison: notebook as written vs. incremental fixes vs. baselines.

Model selection only ever looks at validation data; test is scored once per finished run.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
import copy
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
R = str(HERE.parent) + "/"
CAT = ["workclass", "marital_status", "occupation", "relationship", "race", "gender", "native_country"]
NUM = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]


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


def nb_prep(tr, te, seed=None):
    """Replicates the notebook's numbers: positional split, mode imputation, scaler fit on train+val."""
    tr, te = tr.copy(), te.copy()
    for d in (tr, te):
        d[CAT] = d[CAT].replace("?", np.nan)
    for d in (tr, te):  # separately-fit imputers, same as notebook
        for c in CAT:
            d[c] = d[c].fillna(d[c].mode()[0])
    sc = StandardScaler().fit(tr[NUM])
    enc = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False).fit(tr[CAT])
    f = lambda d: np.hstack([sc.transform(d[NUM]), enc.transform(d[CAT])]).astype(np.float32)
    X, Xt = f(tr), f(te)
    n = int(0.8 * len(X))
    return X[:n], tr.y.values[:n], X[n:], tr.y.values[n:], Xt, te.y.values


def fixed_prep(tr, te, seed, fnlwgt=False):
    """Stratified random split; '?' kept as a category; log1p capital; everything fit on the train part only."""
    tr, te = tr.copy(), te.copy()
    num = NUM + (["fnlwgt"] if fnlwgt else [])
    for d in (tr, te):
        d[CAT] = d[CAT].replace("?", "Unknown")
        d["capital_gain"] = np.log1p(d["capital_gain"])
        d["capital_loss"] = np.log1p(d["capital_loss"])
    a, b = train_test_split(tr, test_size=0.2, stratify=tr.y, random_state=seed)
    ct = ColumnTransformer([("n", StandardScaler(), num),
                            ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)]).fit(a)
    f = lambda d: ct.transform(d).astype(np.float32)
    return f(a), a.y.values, f(b), b.y.values, f(te), te.y.values


def net(n, logits):
    layers, prev = [], n
    for h, p in [(128, .4), (64, .3), (32, .2)]:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    if not logits:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def fit(data, seed, deep_copy, lr=.01, logits=False, wd=0., sched=False, max_epochs=50, patience=10):
    Xa, ya, Xb, yb, Xt, yt = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    T = lambda x: torch.tensor(x, dtype=torch.float32)
    dl = DataLoader(TensorDataset(T(Xa), T(ya).view(-1, 1)), batch_size=64, shuffle=True)
    Xb_, yb_ = T(Xb), T(yb).view(-1, 1)
    m = net(Xa.shape[1], logits)
    crit = nn.BCEWithLogitsLoss() if logits else nn.BCELoss()
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    s = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=3) if sched else None
    best, best_ep, cnt, state = float("inf"), 0, 0, None
    hist = []
    for ep in range(max_epochs):
        m.train()
        for x, y in dl:
            loss = crit(m(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = crit(m(Xb_), yb_).item()
        hist.append(vl)
        if s:
            s.step(vl)
        if vl < best - 1e-5:
            best, best_ep, cnt = vl, ep + 1, 0
            # This line is the notebook's bug when deep_copy=False
            state = copy.deepcopy(m.state_dict()) if deep_copy else m.state_dict().copy()
        else:
            cnt += 1
            if cnt >= patience:
                break
    ended = ep + 1
    m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        restored_vl = crit(m(Xb_), yb_).item()
        o = m(T(Xt)).view(-1)
        p = (torch.sigmoid(o) if logits else o).numpy()
    r = metrics(yt, p)
    r.update(stop_epoch=ended, best_epoch=best_ep, best_val=best, restored_val=restored_vl, last_val=hist[-1])
    return r


def metrics(y, p):
    h = (p > .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, h).ravel()
    return dict(acc=100 * accuracy_score(y, h), auc=roc_auc_score(y, p) if len(np.unique(p)) > 1 else .5,
                f1=f1_score(y, h, zero_division=0), prec=precision_score(y, h, zero_division=0),
                rec=recall_score(y, h, zero_division=0), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


tr, te = load()
SEEDS = [0, 1, 2, 3, 4]
out = {}


def run(name, prep, **kw):
    rows = []
    for s in SEEDS:
        rows.append(fit(prep(tr, te, s), s, **kw))
        print(name, s, {k: round(v, 4) for k, v in rows[-1].items()}, flush=True)
    out[name] = rows


# Baselines (fit on the training part only so they get the same data as the nets)
Xa, ya, Xb, yb, Xt, yt = fixed_prep(tr, te, 0)
out["majority"] = [metrics(yt, DummyClassifier(strategy="most_frequent").fit(Xa, ya).predict_proba(Xt)[:, 1])]
out["logreg"] = [metrics(yt, LogisticRegression(max_iter=2000).fit(Xa, ya).predict_proba(Xt)[:, 1])]
out["hgb"] = [metrics(yt, HistGradientBoostingClassifier(random_state=0).fit(Xa, ya).predict_proba(Xt)[:, 1])]
print({k: out[k] for k in ("majority", "logreg", "hgb")}, flush=True)

run("A_notebook_as_written", nb_prep, deep_copy=False)
run("B_A+deepcopy", nb_prep, deep_copy=True)
run("C_B+lr1e-3", nb_prep, deep_copy=True, lr=1e-3)
run("D_full_fixes", fixed_prep, deep_copy=True, lr=1e-3, logits=True, wd=1e-4, sched=True, max_epochs=100, patience=15)
run("E_D+fnlwgt", lambda a, b, s: fixed_prep(a, b, s, fnlwgt=True), deep_copy=True, lr=1e-3, logits=True, wd=1e-4,
    sched=True, max_epochs=100, patience=15)

json.dump(out, open(RESULTS / "compare_versions.json", "w"), indent=1)
print("\nSUMMARY (mean +/- sd over seeds)")
for k, rows in out.items():
    d = pd.DataFrame(rows)
    print(f"{k:24s} acc {d.acc.mean():6.2f}+/-{d.acc.std() if len(d)>1 else 0:.2f}  auc {d.auc.mean():.4f}  "
          f"f1 {d.f1.mean():.4f}  prec {d.prec.mean():.3f}  rec {d.rec.mean():.3f}"
          + (f"  stop {d.stop_epoch.mean():.0f} best {d.best_epoch.mean():.0f} bestval {d.best_val.mean():.4f} "
             f"restoredval {d.restored_val.mean():.4f}" if "stop_epoch" in d else ""))
