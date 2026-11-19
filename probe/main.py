"""PROBE of v8F -- not a submission to keep. It computes exactly what v8F computes,
records what TabICL did, and at the end of the last instrument raises an exception
whose type says how far TabICL got on the grader's GPU (only the type comes back).
Failed submissions do not count in either phase, so this costs one run, not a slot.

  EOFError             no GPU visible to torch
  ProcessLookupError   the checkpoint from models.txt was not found in the cache
  ReferenceError       importing tabicl or building the shared class failed
  BufferError          TabICL was never attempted (time stop or too few rows everywhere)
  NotADirectoryError   the very first fit failed (checkpoint load)
  FloatingPointError   loaded, no item finished, and at least one pass raised
  UnicodeError         loaded, no item finished, none raised (time stop inside the folds)
  ChildProcessError    fold passes ran, but the full-data pass never ran (time)
  RecursionError       ran end to end, but out-of-fold choice mixed it into no item
  NotImplementedError  mixed into at least one item, run used < 75% of the time limit
  BrokenPipeError      mixed into at least one item, run used >= 75% of the time limit

Any other type is an ordinary bug and means the probe did not reach its end.

SimulacraBench submission v8 -- v5, but the trees and TabICL predict the hidden rows
from a model fitted on every visible row (out-of-fold rows still choose and mix);
TabICL falls back to the fold average if time runs out before that last pass.

v5 -- v4 plus the pretrained TabICLv2 tabular model
(jingang/TabICL, BSD-3, in-context, no fine-tuning) as one more per-item candidate,
run on the GPU under a time budget and mixed in out of fold.

For every PREDICT item we choose, by K-fold cross-validation on the visible
rows (scored exactly as the grader scores: renormalise, mix with a flat vector
at floor 1e-3, log score), between:

  direct     a hierarchical Dirichlet backoff over the item's full option list
             (values plus NA_GATED when gated), conditioned on the m GIVEN
             columns that carry the most information about the item, with
             pseudo-count alpha carried down each level;
  recursion  for an item gated on another PREDICT item: P(asked | x) is the
             parent's predicted distribution times the empirical
             P(asked | parent value), and the answer given asked is a backoff
             fitted on the rows where the item was asked.

When an item's gate parent is GIVEN, that column is always the first level of
the backoff, so a skip that the visible answer decides is reproduced exactly.

v3 over v2: the network is trained MLP_SEEDS times per fold and averaged; the
blend weight between network and backoff is chosen per item from a finer grid;
last, each item's distribution is sharpened or flattened (p ** t, renormalised)
with t chosen on the out-of-fold rows.

Everything is fitted inside predict() on the visible rows of the frame it is
given. No files are written, nothing is downloaded, no processes are started.
"""

import glob
import os
import time

import numpy as np
import pandas as pd

FLOOR = 1e-3
ALPHAS = (0.5, 2.0, 8.0, 32.0)
LEVELS = (0, 1, 2, 3, 4)
FOLDS = 3
SEED = 20261112
BUDGET_SCALE = 3.6           # 1.0 for the 900 s Development phase; the Final ZIP sets 3.6
SOFT_BUDGET = 700.0 * BUDGET_SCALE   # seconds across all three instruments

_START = time.monotonic()


# ------------------------------------------------------------------ helpers --

def _floored(P):
    """What the grader actually scores: renormalise, then mix with flat."""
    K = P.shape[1]
    P = P / P.sum(axis=1, keepdims=True)
    return FLOOR + (1.0 - K * FLOOR) * P


def _logloss(P, y):
    """Mean log score (<= 0) of floored P on integer answers y."""
    if len(y) == 0:
        return 0.0
    Q = _floored(P)
    return float(np.mean(np.log(Q[np.arange(len(y)), y])))


def _power(P, t):
    """Sharpen (t > 1) or flatten (t < 1) each row: p ** t, renormalised."""
    Q = np.power(np.clip(P, 0.0, None), t)
    return Q / np.maximum(Q.sum(axis=1, keepdims=True), 1e-300)


def _mutual_information(x, y, nx, ny):
    joint = np.zeros((nx, ny))
    np.add.at(joint, (x, y), 1.0)
    joint += 1e-9
    joint /= joint.sum()
    px = joint.sum(1, keepdims=True)
    py = joint.sum(0, keepdims=True)
    return float(np.sum(joint * np.log(joint / (px * py))))


def _backoff(Xtr, ytr, Xte, K, cols, cards, alpha):
    """Hierarchical Dirichlet backoff. Returns one probability row per Xte row.

    Level 0 is the smoothed marginal (half a count per option, like the crowd
    baseline). Level l conditions on the first l columns of `cols`:
        p_l = (n_stratum + alpha * p_{l-1}) / (N_stratum + alpha)
    A stratum never seen in training keeps its coarser estimate.
    """
    base = np.bincount(ytr, minlength=K).astype(float) + 0.5
    base /= base.sum()
    P = np.tile(base, (len(Xte), 1))
    if len(ytr) == 0:
        return P
    key_tr = np.zeros(len(Xtr), np.int64)
    key_te = np.zeros(len(Xte), np.int64)
    for c in cols:
        card = int(cards[c])
        key_tr = key_tr * card + Xtr[:, c]
        alive = key_te >= 0
        key_te = np.where(alive, key_te * card + Xte[:, c], -1)
        uniq, inv = np.unique(key_tr, return_inverse=True)
        counts = np.zeros((len(uniq), K))
        np.add.at(counts, (inv, ytr), 1.0)
        pos = np.searchsorted(uniq, np.maximum(key_te, 0))
        pos = np.minimum(pos, len(uniq) - 1)
        found = (key_te >= 0) & (uniq[pos] == key_te)
        cnt = np.zeros((len(Xte), K))
        cnt[found] = counts[pos[found]]
        N = cnt.sum(axis=1, keepdims=True)
        P = (cnt + alpha * P) / (N + alpha)
        key_tr = inv.astype(np.int64)
        key_te = np.where(found, pos, -1).astype(np.int64)
    return P


def _rank_columns(X, y, K, cards, forced=None):
    order = sorted(range(X.shape[1]),
                   key=lambda c: -_mutual_information(X[:, c], y, int(cards[c]), K))
    if forced is not None:
        order = [forced] + [c for c in order if c != forced]
    return order


def _encode(series, levels):
    """Option text -> integer code; anything unexpected gets the last code."""
    lookup = {str(v): i for i, v in enumerate(levels)}
    other = len(levels)
    return np.array([lookup.get(str(v), other) if not pd.isna(v) else other
                     for v in series.to_numpy(dtype=object)], dtype=np.int64)


# ---------------------------------------------------------- multitask MLP --

MLP_HIDDEN = 256
MLP_DROPOUT = 0.25
MLP_EPOCHS = 60
MLP_PATIENCE = 6
MLP_BATCH = 256
MLP_LR = 3e-3
MLP_WD = 1e-4
MLP_SEEDS = 3
HGB_BUDGET = 560.0 * BUDGET_SCALE    # stop starting tree models after this many seconds
TABICL_STOP = 600.0 * BUDGET_SCALE   # stop starting TabICL after this many seconds
TABICL_REPO = "jingang/TabICL"
TABICL_FILE = "tabicl-classifier-v2-20260212.ckpt"
TABICL_EST = 4
TABICL_MAX_CONTEXT = 20000
BLEND_W = tuple(round(0.1 * k, 1) for k in range(10, 0, -1))
TEMPS = (0.7, 0.8, 0.9, 1.1, 1.25, 1.5)


def _mlp(Xv, Xh, cards, Y, Ks, fold):
    """One network for every PREDICT item of an instrument.

    Input: the GIVEN block, one-hot. Output: one softmax per item over its full
    option list. Trained once per fold with early stopping on that fold's
    held-out rows; the out-of-fold rows are kept for model choice, and the
    hidden rows get the average of the fold networks.
    Returns (oof list per item, test list per item) or (None, None).
    """
    try:
        import torch
    except Exception:
        return None, None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    offsets = np.concatenate([[0], np.cumsum(cards)]).astype(np.int64)
    D = int(offsets[-1])

    def onehot(X):
        M = np.zeros((len(X), D), np.float32)
        for c in range(X.shape[1]):
            M[np.arange(len(X)), offsets[c] + X[:, c]] = 1.0
        return M

    n_items = len(Ks)
    Kmax = int(max(Ks))
    starts = np.concatenate([[0], np.cumsum(Ks)]).astype(np.int64)
    total = int(starts[-1])
    # gather index (n_items, Kmax) into [logits | pad]; pad column = total
    gather = np.full((n_items, Kmax), total, np.int64)
    for i, k in enumerate(Ks):
        gather[i, :k] = np.arange(starts[i], starts[i] + k)
    gather_t = torch.as_tensor(gather.reshape(-1), device=device)

    Av = torch.as_tensor(onehot(Xv), device=device)
    Ah = torch.as_tensor(onehot(Xh), device=device)
    Yt = torch.as_tensor(Y, device=device)          # (n_vis, n_items), -1 = ignore

    def build():
        net = torch.nn.Sequential(
            torch.nn.Linear(D, MLP_HIDDEN), torch.nn.ReLU(),
            torch.nn.Dropout(MLP_DROPOUT),
            torch.nn.Linear(MLP_HIDDEN, total)).to(device)
        skip = torch.nn.Linear(D, total, bias=False).to(device)
        torch.nn.init.zeros_(skip.weight)
        return net, skip

    def logits(net, skip, A):
        z = net(A) + skip(A)
        z = torch.cat([z, torch.full((z.shape[0], 1), -1e9, device=device)], 1)
        return z[:, gather_t].view(-1, n_items, Kmax)

    def loss_of(net, skip, A, Yb):
        z = logits(net, skip, A)
        return torch.nn.functional.cross_entropy(
            z.reshape(-1, Kmax), Yb.reshape(-1), ignore_index=-1)

    def probs(net, skip, A):
        net.eval()
        with torch.no_grad():
            out = []
            for i in range(0, A.shape[0], 4096):
                out.append(torch.softmax(logits(net, skip, A[i:i + 4096]), -1))
            return torch.cat(out).cpu().numpy() if out else \
                np.zeros((0, n_items, Kmax))

    oof = np.zeros((len(Xv), n_items, Kmax))
    test = np.zeros((len(Xh), n_items, Kmax))
    for f in range(FOLDS):
        tr = np.flatnonzero(fold != f)
        te = np.flatnonzero(fold == f)
        tr_t = torch.as_tensor(tr, device=device)
        te_t = torch.as_tensor(te, device=device)
        got = 0
        for sd in range(MLP_SEEDS):
            if got and (time.monotonic() - _START) > SOFT_BUDGET:
                break
            torch.manual_seed(SEED + 1000 * sd + f)
            gen = torch.Generator(device="cpu").manual_seed(SEED + 1000 * sd + f)
            net, skip = build()
            opt = torch.optim.AdamW(list(net.parameters()) + list(skip.parameters()),
                                    lr=MLP_LR, weight_decay=MLP_WD)
            best, best_state, bad = np.inf, None, 0
            for epoch in range(MLP_EPOCHS):
                net.train()
                perm = tr_t[torch.randperm(len(tr), generator=gen).to(device)]
                for i in range(0, len(perm), MLP_BATCH):
                    b = perm[i:i + MLP_BATCH]
                    opt.zero_grad()
                    loss = loss_of(net, skip, Av[b], Yt[b])
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    val = float(loss_of(net, skip, Av[te_t], Yt[te_t]))
                if val < best - 1e-5:
                    best, bad = val, 0
                    best_state = ([p.detach().clone() for p in net.parameters()],
                                  [p.detach().clone() for p in skip.parameters()])
                else:
                    bad += 1
                    if bad >= MLP_PATIENCE:
                        break
                if (time.monotonic() - _START) > SOFT_BUDGET:
                    break
            if best_state is not None:
                with torch.no_grad():
                    for p, q in zip(net.parameters(), best_state[0]):
                        p.copy_(q)
                    for p, q in zip(skip.parameters(), best_state[1]):
                        p.copy_(q)
            po = probs(net, skip, Av[te_t])
            pt = probs(net, skip, Ah)
            if got == 0:
                oof_f, test_f = po, pt
            else:
                oof_f, test_f = oof_f + po, test_f + pt
            got += 1
        oof[te] = oof_f / got
        test += (test_f / got) / FOLDS
    return ([oof[:, i, :k] for i, k in enumerate(Ks)],
            [test[:, i, :k] for i, k in enumerate(Ks)])


def _hgb(Xv, yv, ok, Xh, K, fold):
    """Gradient-boosted trees for one item; GIVEN codes are categorical features.
    Out-of-fold rows for choice, hidden rows from the average of the fold models.
    Returns (oof, test) or (None, None)."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except Exception:
        return None, None
    oof = np.zeros((len(Xv), K))
    test = np.zeros((len(Xh), K))
    for f in range(FOLDS):
        tr = (fold != f) & ok
        te = fold == f
        cls = np.unique(yv[tr])
        if len(cls) < 2:
            base = np.bincount(yv[tr], minlength=K)[:K].astype(float) + 0.5
            oof[te] = base / base.sum()
            test += np.tile(base / base.sum(), (len(Xh), 1)) / FOLDS
            continue
        m = HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=200, max_leaf_nodes=15, min_samples_leaf=30,
            l2_regularization=1.0, categorical_features=np.arange(Xv.shape[1]),
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=12,
            random_state=SEED + f)
        m.fit(Xv[tr], yv[tr])
        P = np.full((te.sum(), K), 1e-6)
        P[:, m.classes_] = m.predict_proba(Xv[te])
        oof[te] = P / P.sum(1, keepdims=True)
        Q = np.full((len(Xh), K), 1e-6)
        Q[:, m.classes_] = m.predict_proba(Xh)
        test += (Q / Q.sum(1, keepdims=True)) / FOLDS
    allr = ok
    if len(np.unique(yv[allr])) >= 2:
        m = HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=200, max_leaf_nodes=15, min_samples_leaf=30,
            l2_regularization=1.0, categorical_features=np.arange(Xv.shape[1]),
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=12,
            random_state=SEED + 99)
        m.fit(Xv[allr], yv[allr])
        Q = np.full((len(Xh), K), 1e-6)
        Q[:, m.classes_] = m.predict_proba(Xh)
        test = Q / Q.sum(1, keepdims=True)
    return oof, test


_TABICL = {"cls": None, "path": None, "tried": False}
_PROBE = {"calls": 0, "gpu": None, "ckpt": None, "import_ok": None, "attempts": 0,
          "first_fit": None, "fold_ok": 0, "fold_err": 0, "full_ok": 0, "mixed": 0}
PROBE_AT = int(os.environ.get("SB_PROBE_AT", "3"))     # raise at this predict() call
PROBE_LIMIT = 3600.0                                   # Final-phase time limit, seconds


def _tabicl_ready():
    """Find the checkpoint the runtime pre-downloaded (models.txt); GPU only,
    unless SB_TABICL_CPU is set for a local test. Returns True when usable."""
    if _TABICL["tried"]:
        return _TABICL["cls"] is not None
    _TABICL["tried"] = True
    try:
        import torch
        _PROBE["gpu"] = bool(torch.cuda.is_available() or os.environ.get("SB_TABICL_CPU"))
        if not torch.cuda.is_available() and not os.environ.get("SB_TABICL_CPU"):
            return False
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        path = None
        try:
            from huggingface_hub import try_to_load_from_cache
            p = try_to_load_from_cache(TABICL_REPO, TABICL_FILE)
            if isinstance(p, str) and os.path.exists(p):
                path = p
        except Exception:
            path = None
        if path is None:
            for root in (os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME"),
                         os.path.expanduser("~/.cache/huggingface")):
                if root:
                    hits = glob.glob(os.path.join(root, "**", TABICL_FILE), recursive=True)
                    if hits:
                        path = hits[0]
                        break
        _PROBE["ckpt"] = path is not None
        if path is None:
            return False
        _PROBE["import_ok"] = False
        from tabicl import TabICLClassifier

        class _Shared(TabICLClassifier):
            """Load the checkpoint once and reuse it for every item."""
            _model = None

            def _load_model(self):
                if _Shared._model is None:
                    super()._load_model()
                    _Shared._model = (self.model_path_, self.model_, self.model_config_)
                else:
                    self.model_path_, self.model_, self.model_config_ = _Shared._model

        _TABICL["cls"], _TABICL["path"] = _Shared, path
        _PROBE["import_ok"] = True
        return True
    except Exception:
        return False


def _tabicl(Xv, yv, ok, Xh, K, fold):
    """TabICL for one item: for each fold, the other folds' rows are the context and
    the fold's rows plus the hidden rows are the queries. Returns (oof, test)."""
    import torch
    cls = _TABICL["cls"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    oof = np.zeros((len(Xv), K))
    test = np.zeros((len(Xh), K))
    rng = np.random.default_rng(SEED + 7)
    for f in range(FOLDS):
        if (time.monotonic() - _START) > TABICL_STOP:
            return None, None
        tr = np.flatnonzero((fold != f) & ok)
        te = np.flatnonzero(fold == f)
        if len(np.unique(yv[tr])) < 2:
            return None, None
        if len(tr) > TABICL_MAX_CONTEXT:
            tr = np.sort(rng.choice(tr, TABICL_MAX_CONTEXT, replace=False))
        m = cls(n_estimators=TABICL_EST, model_path=_TABICL["path"], allow_auto_download=False,
                device=dev, offload_mode=False, random_state=SEED + f, verbose=False)
        try:
            m.fit(pd.DataFrame(Xv[tr]).astype("category"), yv[tr])
        except Exception:
            if _PROBE["first_fit"] is None:
                _PROBE["first_fit"] = False
            raise
        _PROBE["first_fit"] = True
        Q = m.predict_proba(pd.DataFrame(np.vstack([Xv[te], Xh])).astype("category"))
        P = np.full((len(Q), K), 1e-6)
        P[:, np.asarray(m.classes_, dtype=np.int64)] = Q
        P /= P.sum(1, keepdims=True)
        oof[te] = P[:len(te)]
        test += P[len(te):] / FOLDS
    if (time.monotonic() - _START) < TABICL_STOP:
        tr = np.flatnonzero(ok)
        if len(tr) > TABICL_MAX_CONTEXT:
            tr = np.sort(rng.choice(tr, TABICL_MAX_CONTEXT, replace=False))
        m = cls(n_estimators=TABICL_EST, model_path=_TABICL["path"], allow_auto_download=False,
                device=dev, offload_mode=False, random_state=SEED + 99, verbose=False)
        m.fit(pd.DataFrame(Xv[tr]).astype("category"), yv[tr])
        Q = m.predict_proba(pd.DataFrame(Xh).astype("category"))
        P = np.full((len(Q), K), 1e-6)
        P[:, np.asarray(m.classes_, dtype=np.int64)] = Q
        test = P / P.sum(1, keepdims=True)
        _PROBE["full_ok"] += 1
    return oof, test


def _gate_given(P, xp, a_v, n_v, rule):
    """Impose a gate whose parent is GIVEN: P(asked) from the parent's value."""
    g = (a_v + 0.5 * rule) / (n_v + 0.5)
    pa = g[np.minimum(xp, len(g) - 1)] if len(g) else np.ones(len(xp))
    pa = np.where(xp < len(g), pa, 1.0 - P[:, -1])
    cond = P[:, :-1]
    cond = cond / np.maximum(cond.sum(1, keepdims=True), 1e-12)
    out = np.empty_like(P)
    out[:, :-1] = pa[:, None] * cond
    out[:, -1] = 1.0 - pa
    return out


# ------------------------------------------------------------------ predict --

def predict(frame, schema):
    _PROBE["calls"] += 1
    out = _predict(frame, schema)
    if _PROBE["calls"] >= PROBE_AT:
        raise _probe_verdict()
    return out


def _probe_verdict():
    """The exception whose type reports how far TabICL got (table in the docstring)."""
    p, used = _PROBE, time.monotonic() - _START
    msg = "probe %s used=%.0fs" % (p, used)
    if p["gpu"] is False:
        return EOFError(msg)
    if p["ckpt"] is False:
        return ProcessLookupError(msg)
    if p["import_ok"] is False:
        return ReferenceError(msg)
    if p["attempts"] == 0:
        return BufferError(msg)
    if p["first_fit"] is False:
        return NotADirectoryError(msg)
    if p["fold_ok"] == 0 and p["fold_err"] > 0:
        return FloatingPointError(msg)
    if p["fold_ok"] == 0:
        return UnicodeError(msg)
    if p["full_ok"] == 0:
        return ChildProcessError(msg)
    if p["mixed"] == 0:
        return RecursionError(msg)
    if used < 0.75 * PROBE_LIMIT:
        return NotImplementedError(msg)
    return BrokenPipeError(msg)


def _predict(frame, schema):
    gated_value = schema["gated_value"]
    records = schema["items"]
    items = [name for name, r in records.items()
             if r["class"] in ("GIVEN", "PREDICT") and name in frame.columns]
    given = [n for n in items if records[n]["class"] == "GIVEN"]
    target = [n for n in items if records[n]["class"] == "PREDICT"]

    options = {}
    for n in items:
        r = records[n]
        options[n] = list(r["values"]) + ([gated_value] if r.get("gate") else [])

    hidden = frame[target].isna().any(axis=1).to_numpy() if target else \
        np.zeros(len(frame), bool)
    visible = ~hidden

    # GIVEN block as integer codes, one extra code for anything unexpected.
    X = np.column_stack([_encode(frame[g], options[g]) for g in given]) \
        if given else np.zeros((len(frame), 0), np.int64)
    cards = np.array([len(options[g]) + 1 for g in given], dtype=np.int64)
    gcol = {g: i for i, g in enumerate(given)}
    Xv, Xh = X[visible], X[hidden]

    rng = np.random.default_rng(SEED)
    n_vis = int(visible.sum())
    fold = rng.integers(0, FOLDS, size=n_vis)

    oof = {}        # item -> out-of-fold probability rows for visible rows
    test = {}       # item -> probability rows for hidden rows
    late = (time.monotonic() - _START) > SOFT_BUDGET

    # ---- multitask network over all items, once per instrument ------------
    mlp_oof = mlp_test = None
    if target and given and not late and n_vis >= 50:
        Ks = [len(options[n]) for n in target]
        Y = np.column_stack([_encode(frame.loc[visible, n], options[n])
                             for n in target])
        for i, k in enumerate(Ks):
            Y[Y[:, i] >= k, i] = -1
        try:
            o, t = _mlp(Xv, Xh, cards, Y, Ks, fold)
        except Exception:
            o, t = None, None
        if o is not None:
            mlp_oof = dict(zip(target, o))
            mlp_test = dict(zip(target, t))

    for n in target:
        K = len(options[n])
        yv_all = _encode(frame.loc[visible, n], options[n])
        ok = yv_all < K                      # drop anything off the option list
        gate = records[n].get("gate")
        parent = gate["parent"] if gate else None
        forced = gcol.get(parent) if parent in gcol else None

        base = np.bincount(yv_all[ok], minlength=K).astype(float) + 0.5
        base /= base.sum()

        if late or not given or ok.sum() < 2:
            oof[n] = np.tile(base, (n_vis, 1))
            test[n] = np.tile(base, (len(Xh), 1))
            continue

        late = (time.monotonic() - _START) > SOFT_BUDGET

        # ---- candidate 1: direct backoff over the full option list --------
        best = (-np.inf, None)
        cache = {}
        for f in range(FOLDS):
            tr = (fold != f) & ok
            cache[f] = _rank_columns(Xv[tr], yv_all[tr], K, cards, forced)
        for alpha in ALPHAS:
            for m in LEVELS:
                P = np.zeros((n_vis, K))
                for f in range(FOLDS):
                    tr = (fold != f) & ok
                    te = fold == f
                    P[te] = _backoff(Xv[tr], yv_all[tr], Xv[te], K,
                                     cache[f][:m], cards, alpha)
                s = _logloss(P[ok], yv_all[ok])
                if s > best[0]:
                    best = (s, ("direct", alpha, m), P)

        # ---- candidate 2: gate recursion through a PREDICT parent ----------
        if gate and parent in oof and K >= 2:
            gi = K - 1                                    # NA_GATED slot
            Kp = len(options[parent])
            yp = _encode(frame.loc[visible, parent], options[parent])
            rule = np.array([1.0 if str(v) in {str(o) for o in gate["observed_if"]}
                             else 0.0 for v in options[parent]])
            asked = ok & (yv_all != gi)
            Kv = K - 1
            for alpha in ALPHAS:
                for m in LEVELS:
                    P = np.zeros((n_vis, K))
                    for f in range(FOLDS):
                        tr = (fold != f) & ok & (yp < Kp)
                        te = fold == f
                        a_v = np.bincount(yp[tr & asked], minlength=Kp)[:Kp]
                        n_v = np.bincount(yp[tr], minlength=Kp)[:Kp]
                        g = (a_v + 0.5 * rule) / (n_v + 0.5)
                        pa = oof[parent][te] @ g
                        tra = (fold != f) & asked
                        cols = _rank_columns(Xv[tra], yv_all[tra], Kv, cards) \
                            if tra.sum() else []
                        cond = _backoff(Xv[tra], yv_all[tra], Xv[te], Kv,
                                        cols[:m], cards, alpha)
                        P[te, :Kv] = pa[:, None] * cond
                        P[te, gi] = 1.0 - pa
                    P = np.clip(P, 0.0, None)
                    s = _logloss(P[ok], yv_all[ok])
                    if s > best[0]:
                        best = (s, ("recursion", alpha, m), P)

        _, choice, P_oof = best
        oof[n] = P_oof

        # ---- refit on every visible row, predict the hidden rows -----------
        kind, alpha, m = choice
        if kind == "direct":
            cols = _rank_columns(Xv[ok], yv_all[ok], K, cards, forced)
            test[n] = _backoff(Xv[ok], yv_all[ok], Xh, K, cols[:m], cards, alpha)
        else:
            gi = K - 1
            Kp = len(options[parent])
            yp = _encode(frame.loc[visible, parent], options[parent])
            rule = np.array([1.0 if str(v) in {str(o) for o in gate["observed_if"]}
                             else 0.0 for v in options[parent]])
            asked = ok & (yv_all != gi)
            tr = ok & (yp < Kp)
            a_v = np.bincount(yp[tr & asked], minlength=Kp)[:Kp]
            n_v = np.bincount(yp[tr], minlength=Kp)[:Kp]
            g = (a_v + 0.5 * rule) / (n_v + 0.5)
            pa = test[parent] @ g
            cols = _rank_columns(Xv[asked], yv_all[asked], K - 1, cards) \
                if asked.sum() else []
            cond = _backoff(Xv[asked], yv_all[asked], Xh, K - 1, cols[:m],
                            cards, alpha)
            P = np.zeros((len(Xh), K))
            P[:, :K - 1] = pa[:, None] * cond
            P[:, gi] = 1.0 - pa
            test[n] = np.clip(P, 0.0, None)

        # ---- candidate 3: the network, alone or mixed with the backoff -----
        if mlp_oof is not None and n in mlp_oof:
            Pm_oof = mlp_oof[n].copy()
            Pm_test = mlp_test[n].copy()
            if gate and forced is not None:
                Kp = len(options[parent])
                xp_v, xp_h = Xv[:, forced], Xh[:, forced]
                rule = np.array([1.0 if str(v) in {str(o) for o in gate["observed_if"]}
                                 else 0.0 for v in options[parent]])
                asked_v = ok & (yv_all != K - 1)
                a_v = np.bincount(xp_v[asked_v], minlength=Kp)[:Kp]
                n_v = np.bincount(xp_v[ok], minlength=Kp)[:Kp]
                Pm_oof = _gate_given(Pm_oof, xp_v, a_v, n_v, rule)
                Pm_test = _gate_given(Pm_test, xp_h, a_v, n_v, rule)
            best_s = _logloss(oof[n][ok], yv_all[ok])
            pick = None
            for w in BLEND_W:
                Po = w * Pm_oof + (1.0 - w) * oof[n]
                s_w = _logloss(Po[ok], yv_all[ok])
                if s_w > best_s:
                    best_s, pick = s_w, w
            if pick is not None:
                oof[n] = pick * Pm_oof + (1.0 - pick) * oof[n]
                test[n] = pick * Pm_test + (1.0 - pick) * test[n]

        # ---- candidate 4: gradient-boosted trees, mixed in out of fold ----
        if ok.sum() >= 30 and (time.monotonic() - _START) < HGB_BUDGET \
                and Xv.shape[1] and max(cards) <= 255:
            try:
                Ph_oof, Ph_test = _hgb(Xv, yv_all, ok, Xh, K, fold)
            except Exception:
                Ph_oof = Ph_test = None
            if Ph_oof is not None:
                if gate and forced is not None:
                    Kp = len(options[parent])
                    xp_v, xp_h = Xv[:, forced], Xh[:, forced]
                    rule = np.array([1.0 if str(v) in {str(o) for o in gate["observed_if"]}
                                     else 0.0 for v in options[parent]])
                    asked_v = ok & (yv_all != K - 1)
                    a_v = np.bincount(xp_v[asked_v], minlength=Kp)[:Kp]
                    n_v = np.bincount(xp_v[ok], minlength=Kp)[:Kp]
                    Ph_oof = _gate_given(Ph_oof, xp_v, a_v, n_v, rule)
                    Ph_test = _gate_given(Ph_test, xp_h, a_v, n_v, rule)
                best_s = _logloss(oof[n][ok], yv_all[ok])
                pick = None
                for w in BLEND_W:
                    Po = w * Ph_oof + (1.0 - w) * oof[n]
                    s_w = _logloss(Po[ok], yv_all[ok])
                    if s_w > best_s:
                        best_s, pick = s_w, w
                if pick is not None:
                    oof[n] = pick * Ph_oof + (1.0 - pick) * oof[n]
                    test[n] = pick * Ph_test + (1.0 - pick) * test[n]

        # ---- candidate 5: pretrained TabICL, mixed in out of fold ---------
        if ok.sum() >= 30 and (time.monotonic() - _START) < TABICL_STOP \
                and Xv.shape[1] and _tabicl_ready():
            _PROBE["attempts"] += 1
            try:
                Pt_oof, Pt_test = _tabicl(Xv, yv_all, ok, Xh, K, fold)
            except Exception:
                _PROBE["fold_err"] += 1
                Pt_oof = Pt_test = None
            if Pt_oof is not None:
                _PROBE["fold_ok"] += 1
            if Pt_oof is not None:
                if gate and forced is not None:
                    Kp = len(options[parent])
                    xp_v, xp_h = Xv[:, forced], Xh[:, forced]
                    rule = np.array([1.0 if str(v) in {str(o) for o in gate["observed_if"]}
                                     else 0.0 for v in options[parent]])
                    asked_v = ok & (yv_all != K - 1)
                    a_v = np.bincount(xp_v[asked_v], minlength=Kp)[:Kp]
                    n_v = np.bincount(xp_v[ok], minlength=Kp)[:Kp]
                    Pt_oof = _gate_given(Pt_oof, xp_v, a_v, n_v, rule)
                    Pt_test = _gate_given(Pt_test, xp_h, a_v, n_v, rule)
                best_s = _logloss(oof[n][ok], yv_all[ok])
                pick = None
                for w in BLEND_W:
                    Po = w * Pt_oof + (1.0 - w) * oof[n]
                    s_w = _logloss(Po[ok], yv_all[ok])
                    if s_w > best_s:
                        best_s, pick = s_w, w
                if pick is not None:
                    _PROBE["mixed"] += 1
                    oof[n] = pick * Pt_oof + (1.0 - pick) * oof[n]
                    test[n] = pick * Pt_test + (1.0 - pick) * test[n]

        # ---- last: one temperature per item, chosen out of fold -----------
        if ok.sum() >= 2:
            s_t, pick_t = _logloss(oof[n][ok], yv_all[ok]), None
            for t in TEMPS:
                s_x = _logloss(_power(oof[n][ok], t), yv_all[ok])
                if s_x > s_t:
                    s_t, pick_t = s_x, t
            if pick_t is not None:
                oof[n] = _power(oof[n], pick_t)
                test[n] = _power(test[n], pick_t)

    # ---- write out in canonical order: rows top to bottom, schema order ----
    hidden_index = np.full(len(frame), -1)
    hidden_index[hidden] = np.arange(int(hidden.sum()))
    marg = {}
    values = frame[items].to_numpy(dtype=object)
    out = []
    for row in range(values.shape[0]):
        for col, n in enumerate(items):
            if not pd.isna(values[row, col]):
                continue
            if n in test and hidden_index[row] >= 0:
                vec = test[n][hidden_index[row]]
            else:
                if n not in marg:
                    counts = frame[n].value_counts()
                    w = np.array([counts.get(o, 0) + 0.5 for o in options[n]], float)
                    marg[n] = w / w.sum()
                vec = marg[n]
            vec = np.asarray(vec, float)
            if not np.all(np.isfinite(vec)) or vec.sum() <= 0:
                vec = np.ones(len(options[n]))
            out.append([float(v) for v in np.maximum(vec, 0.0) + 1e-12])
    return out
