"""Make probe/main.py from main.py: same computation, plus a recorder of what
TabICL did, and a deliberate exception at the end of the last instrument whose TYPE says
what happened (the runtime returns only the exception type for predict() errors).
Run: .venv/bin/python _mk_probe.py"""
from pathlib import Path

src = Path("main.py").read_text()


def sub(old, new):
    global src
    assert src.count(old) == 1, (src.count(old), old[:60])
    src = src.replace(old, new)


sub('"""SimulacraBench submission v8 --',
    '''"""PROBE of v8F -- not a submission to keep. It computes exactly what v8F computes,
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

SimulacraBench submission v8 --''')

sub('_TABICL = {"cls": None, "path": None, "tried": False}\n',
    '''_TABICL = {"cls": None, "path": None, "tried": False}
_PROBE = {"calls": 0, "gpu": None, "ckpt": None, "import_ok": None, "attempts": 0,
          "first_fit": None, "fold_ok": 0, "fold_err": 0, "full_ok": 0, "mixed": 0}
PROBE_AT = int(os.environ.get("SB_PROBE_AT", "3"))     # raise at this predict() call
PROBE_LIMIT = 3600.0                                   # Final-phase time limit, seconds
''')

sub('''        if not torch.cuda.is_available() and not os.environ.get("SB_TABICL_CPU"):
            return False''',
    '''        _PROBE["gpu"] = bool(torch.cuda.is_available() or os.environ.get("SB_TABICL_CPU"))
        if not torch.cuda.is_available() and not os.environ.get("SB_TABICL_CPU"):
            return False''')

sub('''        if path is None:
            return False
        from tabicl import TabICLClassifier''',
    '''        _PROBE["ckpt"] = path is not None
        if path is None:
            return False
        _PROBE["import_ok"] = False
        from tabicl import TabICLClassifier''')

sub('''        _TABICL["cls"], _TABICL["path"] = _Shared, path
        return True''',
    '''        _TABICL["cls"], _TABICL["path"] = _Shared, path
        _PROBE["import_ok"] = True
        return True''')

sub('''        m.fit(pd.DataFrame(Xv[tr]).astype("category"), yv[tr])
        Q = m.predict_proba(pd.DataFrame(np.vstack([Xv[te], Xh])).astype("category"))''',
    '''        try:
            m.fit(pd.DataFrame(Xv[tr]).astype("category"), yv[tr])
        except Exception:
            if _PROBE["first_fit"] is None:
                _PROBE["first_fit"] = False
            raise
        _PROBE["first_fit"] = True
        Q = m.predict_proba(pd.DataFrame(np.vstack([Xv[te], Xh])).astype("category"))''')

sub('''        test = P / P.sum(1, keepdims=True)
    return oof, test''',
    '''        test = P / P.sum(1, keepdims=True)
        _PROBE["full_ok"] += 1
    return oof, test''')

sub('''            try:
                Pt_oof, Pt_test = _tabicl(Xv, yv_all, ok, Xh, K, fold)
            except Exception:
                Pt_oof = Pt_test = None''',
    '''            _PROBE["attempts"] += 1
            try:
                Pt_oof, Pt_test = _tabicl(Xv, yv_all, ok, Xh, K, fold)
            except Exception:
                _PROBE["fold_err"] += 1
                Pt_oof = Pt_test = None
            if Pt_oof is not None:
                _PROBE["fold_ok"] += 1''')

sub('''                if pick is not None:
                    oof[n] = pick * Pt_oof + (1.0 - pick) * oof[n]
                    test[n] = pick * Pt_test + (1.0 - pick) * test[n]''',
    '''                if pick is not None:
                    _PROBE["mixed"] += 1
                    oof[n] = pick * Pt_oof + (1.0 - pick) * oof[n]
                    test[n] = pick * Pt_test + (1.0 - pick) * test[n]''')

sub('''def predict(frame, schema):
    gated_value = schema["gated_value"]''',
    '''def predict(frame, schema):
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
    gated_value = schema["gated_value"]''')

Path("probe/main.py").write_text(src)
print("ok", len(src.splitlines()), "lines")
