# Known traps — read before you start

Seven things that cost hours and teach nothing. They are not part of the assessment, and fixes
are given for all of them.

**Three are already fixed in the code you are given. The rest are live in the code you write.**

| # | Trap | Status |
|---|---|---|
| 1 | `statsmodels` params are an unnamed array | **Live** — bites in your `stage5` bootstrap |
| 2 | `yfinance` MultiIndex columns | Fixed in `stage1`; live if you add your own tickers |
| 3 | `.where().rolling().std()` returns all-NaN | Fixed for realised volatility; live for features you add |
| 4 | Regime labels arbitrary between fits | Fixed in `stage4`. **Live in your bootstrap** — every draw is a fresh fit |
| 5 | Perturbation effect smaller than sampling noise | Half fixed: `score_field()` gives you the averaging; using it is yours |
| 6 | Bootstrap is CPU-bound | **Live** — `stage5` is a stub |
| 7 | Missing data hides in plain sight | A habit to build, not a bug to fix |

**Traps 1, 4 and 6 all land in `src/stage5_uncertainty.py`**, which ships as a stub. Trap 4 is
the dangerous one there: it is silent. Pooling bootstrap draws without re-ordering each fit's
regimes yields a tidy, plausible interval that has quietly mixed different regimes together, and
nothing in your output looks wrong.

---

## 1. `statsmodels` results params are a numpy array, not a named Series

**This is the single most likely place to lose an afternoon.**

```python
res = mod.fit()
res.params["sigma2[0]"]        # IndexError - and this is the obvious thing to try
```

`res.params` is a bare numpy array. The names live separately.

```python
import numpy as np
pmap = dict(zip(res.model.param_names, np.asarray(res.params)))
sigmas = np.sqrt([pmap[f"sigma2[{i}]"] for i in range(config.N_REGIMES)])
```

---

## 2. `yfinance` returns MultiIndex columns even for a single ticker

`df["Close"]` gives you a DataFrame, not a Series, and the shape error surfaces far from
the cause.

```python
df = yf.download("^AXJO", start="2006-01-01", progress=False, auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)
```

---

## 3. `.where().rolling().std()` silently returns an all-NaN column

```python
df["downside_vol"] = df["ret"].where(df["ret"] < 0).rolling(21).std()   # every value NaN
```

`.where()` leaves roughly half the window as NaN, and `rolling().std()` requires a full
window by default. **No error is raised.** You will feed an empty column into your model.

```python
df["downside_vol"] = (df["ret"].where(df["ret"] < 0)
                      .rolling(21, min_periods=5).std() * np.sqrt(252))
```

**Habit worth forming:** print `df.isna().sum()` after building features, every time.

---

## 4. Markov switching regime labels are arbitrary between runs

> **This is the trap that will cost you marks rather than time.** `stage4` handles it for the
> main fit. Your bootstrap in `stage5` refits the model on every draw, and each of those fits
> labels its regimes independently. Re-order **inside the loop**, before pooling. And do not
> order on `sigma2`: with `endog = log_rv` that is the variance of *log* volatility, not
> volatility. Order on empirical realised volatility per assigned regime, as `stage4` does.

Regime "0" in one fit is not regime "0" in the next. Sort by fitted volatility so labels
mean something and are stable.

```python
order = np.argsort(sigmas)                      # low -> high volatility
remap = {int(old): int(new) for new, old in enumerate(order)}
probs.columns = [remap[c] for c in probs.columns]
```

---

## 5. The perturbation effect is smaller than the sampling noise

**Read this before section 8.3.**

One scoring call at temperature 1.0 has a standard deviation of roughly **0.05 to 0.10**.
The effect of changing one phrase in a document is about **the same size**. Measured naively,
you get noise:

| | Mean absolute change |
|---|---|
| Signal edits (meaning reversed) | 0.000 |
| Control edits (meaning preserved) | 0.050 |

The controls moved *more* than the signal. That is not a finding, it is a failed measurement.

**Two changes make the experiment work, and you need both:**

1. **Average several calls per score.** `score_field(text, n=5)` in `stage3_riskvoice.py`
   does this. Noise falls with the square root of the call count.
2. **Perturb the retrieved passage, not the full document.** One phrase changed in 20,000
   characters is diluted to nothing. In ~2,700 characters it is detectable.

Even done properly the effect is modest. Report what you measure.

---

## 6. Bootstrap is CPU-bound — parallelise it

Each Markov switching refit takes about **6 seconds** regardless of settings. Reducing
iterations does not help; it just stops the model converging.

40 draws sequentially is **four minutes**. The draws are independent, so run them across
processes:

```python
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=os.cpu_count() - 2) as ex:
    results = list(ex.map(one_bootstrap_draw, range(n_draws)))
```

**`stage5_uncertainty.py` does NOT do this — it is a stub, and writing the bootstrap is your
job.** 40–60 draws is enough for a 90% interval; you do not need 1,000. Report how many draws
converged against how many you requested, because a quietly-dropped third of them is itself a
finding about model stability.

---

## 7. Ask for `df.isna().sum()` habitually

Three of the traps above produce silent NaN or silently wrong values. None of them raise.
The cheapest defence is printing missingness after every stage, which the supplied stages do.
