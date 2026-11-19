"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's earn rate on
Exophase at the case's target, as five quantiles (5/25/50/75/95%), from the
house's reads of the trophy page up to the case's checkpoint.

The key observation is that an earn rate is a ratio, rate = E / N, and its two
halves move for different reasons in launch week:

- N, the players Exophase tracks, explodes after launch (638 tracked on
  launch night, 3,868 at the first refresh a few hours later, ~10,000 by that
  afternoon) and then its growth decays. Every new player dilutes every rate.
- E, the players who have earned a given trophy, grows only as players
  actually reach that trophy in the game, on that trophy's own timescale:
  an early fishing challenge is earned within hours, a story chapter over
  days, a 250-stunt grind over weeks.

A forecaster that extrapolates the rate itself (the house trend baseline fits
a straight line to logit(rate) against log time) conflates the two: it
overshoots fast-rising trophies, because it cannot see them saturate, and it
misjudges every trophy when the denominator's growth bends. This program
forecasts N and E separately and divides at the end.

How it works, per case:

1. Players joining. From the distinct reads, the join rate J(t) = dN/dt is
   fitted as a power law in time since release, J(t) ~ t^-alpha, on the last
   48 hours of reads (fewer if the history is short). The fit is done on
   increments between distinct reads, so Exophase's frozen stretches (the
   page can sit unchanged for hours between refreshes) contribute nothing
   instead of dragging the fit. N at the target is the last count plus the
   integral of J to the target.
2. Earners per trophy. A player who joined at hour u earns trophy i after a
   delay D ~ Weibull(shape k, scale lambda_i), and only a share p_i of
   players ever earn it by the target week. So, with E_first the earners in
   the first read,
       E_i(t) = E_first + p_i * sum_u J(u) * [F(t-u) - F(t_first-u)],
   where the bracket keeps cohorts that joined before the first read from
   being counted twice. J(u) is the hourly join table from step 1 (the
   opening cohort of N_first players is a lump at u = 0), renormalised so
   its total matches the last tracked count. For each trophy, lambda_i is
   chosen on a grid and p_i solved by least squares through the origin on
   the observed increments; the Weibull shape k is pooled across trophies
   (the grid value with the lowest total fit error). lambda_i is shrunk
   toward the median of trophies with a similar last rate, with the
   shrinkage weight growing in the trophy's observed earner count, so
   trophies almost nobody has earned yet borrow strength from their peers
   instead of chasing noise.
3. Spread. Quantiles are the median on the logit scale plus/minus z_tau
   times sigma, where sigma combines a base term, the trophy's fit
   residual, the uncertainty in N at the target (a dilution shock moves
   every trophy together), and a refresh-lag term: the answer is the page
   as of Exophase's last refresh, which can be hours old, so fast-moving
   trophies get extra width proportional to their recent logit slope.

If the history is frozen (fewer than two distinct reads, as in the practice
cases, where the whole history is one unchanged launch-night page), no trend
can be fitted: the median is shifted down a little (the next refresh
dilutes) and the spread is widened, because a refresh the program has never
seen is exactly what the practice answers contain.

Standard library only, no network, deterministic (no randomness anywhere),
about 0.3 s a case. Reads every <id>.in in MAKEFUN_INPUT_DIR (/input) and
writes <id>.out to MAKEFUN_OUTPUT_DIR (/output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a module shipped beside this file (none is needed here, but the line
# keeps the entry robust to a split layout).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Delay grid (hours) for a trophy's Weibull time-to-earn, and the pooled
# shape values tried. Tuned on simulated launch-week worlds built from the
# real anchors (638 -> 3,868 -> ~10,000 tracked players; live rates of
# 2026-11-19 afternoon): the grid brackets fishing-fast (~hours) to
# platinum-slow (~hundreds of hours).
LAM_GRID = [0.5, 1, 2, 3.5, 6, 10, 16, 26, 42, 68, 110, 175, 280, 450, 700]
K_GRID = [0.7, 1.0, 1.4]

# Join-rate fit: last 48 h of reads (guarded: fewer than 4 positive
# increments falls back to the whole history). Spread terms, in logit units.
JOIN_WINDOW_HOURS = 48.0
MIN_JOIN_INCREMENTS = 4
SIGMA_BASE = 0.22
SIGMA_N = 0.15


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1 - p))


def logistic(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def norm_ppf(p):
    """Standard normal quantile (Acklam's approximation, ~1e-9 accurate)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def mean(xs):
    return sum(xs) / len(xs)


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def _release_origin(case):
    """Hours are counted from the game's release midnight UTC."""
    try:
        rel = case["game"]["release_date"]
        return datetime.fromisoformat(rel).replace(tzinfo=timezone.utc)
    except Exception:
        first = case["history"][0]["at"]
        return _parse_time(first)


def _hours_since(case, origin, text):
    return (_parse_time(text) - origin).total_seconds() / 3600.0


def _distinct_states(case):
    """History compressed to distinct (players_tracked, rates) reads, in order."""
    out = []
    prev = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != prev:
            out.append(read)
            prev = key
    return out


def _fit_join(origin, case, states):
    """Power-law join rate J(t) = exp(log_j0) * t^-alpha from the distinct
    reads of the last JOIN_WINDOW_HOURS (whole history if that window holds
    too few positive increments). Returns (log_j0, alpha, n_last, t_last)."""
    t_cut = _hours_since(case, origin, case["at"])
    window = [s for s in states
              if t_cut - _hours_since(case, origin, s["at"]) <= JOIN_WINDOW_HOURS]
    increments = sum(1 for a, b in zip(window, window[1:])
                     if b["players_tracked"] > a["players_tracked"])
    if len(window) < 2 or increments < MIN_JOIN_INCREMENTS:
        window = states
    xs, ys = [], []
    for a, b in zip(window, window[1:]):
        ta = _hours_since(case, origin, a["at"])
        tb = _hours_since(case, origin, b["at"])
        dn = b["players_tracked"] - a["players_tracked"]
        dt = tb - ta
        if dn > 0 and dt > 0 and tb > 1.0:
            xs.append(math.log((ta + tb) / 2.0))
            ys.append(math.log(dn / dt))
    n_last = states[-1]["players_tracked"]
    t_last = _hours_since(case, origin, states[-1]["at"])
    if len(xs) >= 3 and max(xs) > min(xs):
        mx, my = mean(xs), mean(ys)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sxy / sxx if sxx > 0 else 0.0
        alpha = max(0.0, min(1.2, -slope))
        return my + alpha * mx, alpha, n_last, t_last
    return None, 0.0, n_last, t_last


def _n_forecast(log_j0, alpha, n_last, t_last, t_tgt):
    if log_j0 is None:
        return float(n_last)
    j0 = math.exp(log_j0)
    if abs(alpha - 1.0) < 1e-9:
        add = j0 * math.log(t_tgt / t_last)
    else:
        add = j0 * (t_tgt ** (1 - alpha) - t_last ** (1 - alpha)) / (1 - alpha)
    return n_last + max(add, 0.0)


def _hourly_joins(origin, case, states, log_j0, alpha, t_tgt):
    """Hourly joins J[0..H]: the opening cohort (N_first players) as a lump
    at hour 0, then the fitted power-law join rate, renormalised so the
    cumulative joins up to the last read equal the last tracked count."""
    t_last = _hours_since(case, origin, states[-1]["at"])
    n_first = states[0]["players_tracked"]
    n_last = states[-1]["players_tracked"]
    h_max = int(math.ceil(t_tgt))
    raw = [0.0] * (h_max + 1)
    raw[0] = float(n_first)
    if log_j0 is not None:
        j0 = math.exp(log_j0)
        for u in range(1, h_max + 1):
            raw[u] = j0 * max(u + 0.5, 1.0) ** (-alpha)
    upto = int(math.floor(t_last))
    total = sum(raw[:upto + 1])
    if total > 0:
        scale = n_last / total
        raw = [x * scale for x in raw]
    return raw


def _unit_curve(joins, t_first, h_max, lam, k):
    """Q[h]: earners per unit eventual share by hour h, from the hourly joins
    and a Weibull delay. Cohorts that joined before the first read only
    contribute their earnings after that read."""
    f_delay = [0.0] * (h_max + 1)
    for d in range(h_max + 1):
        f_delay[d] = 1 - math.exp(-((d / lam) ** k)) if d > 0 else 0.0
    first_h = int(round(t_first))
    q = [0.0] * (h_max + 1)
    for h in range(h_max + 1):
        total = 0.0
        for u in range(0, h + 1):
            ju = joins[u]
            if ju <= 0:
                continue
            gain = f_delay[h - u]
            if u <= first_h and 0 <= first_h - u:
                gain -= f_delay[first_h - u]
            if gain > 0:
                total += ju * gain
        q[h] = total
    return q


def _quantiles(median_pct, sigma, levels):
    """Five quantiles around a median on the logit scale."""
    lg = logit(median_pct)
    out = []
    for tau in levels:
        out.append(round4(logistic(lg + norm_ppf(tau) * sigma)))
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def _frozen_forecast(case, levels):
    """No usable trend: the page has never moved. The next refresh dilutes,
    so the median steps down a little and the spread is wide."""
    horizon_days = case["horizon_minutes"] / 1440.0
    sigma = math.sqrt(0.5 ** 2 + (0.5 * math.sqrt(horizon_days)) ** 2)
    last = case["history"][-1]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = logit(rate if rate is not None else 0.0) - 0.30
        rows.append({"id": trophy["id"],
                     "q": _quantiles(logistic(lg), sigma, levels)})
    return {"trophies": rows}


def forecast_case(case):
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    origin = _release_origin(case)
    t_tgt = _hours_since(case, origin, case["target"])
    t_tgt = max(t_tgt, 0.0)
    if not case["history"]:
        rows = [{"id": t["id"], "q": _quantiles(1.0, 1.0, levels)}
                for t in case["trophies"]]
        return {"trophies": rows}
    states = _distinct_states(case)
    if len(states) < 2:
        return _frozen_forecast(case, levels)
    last = case["history"][-1]
    log_j0, alpha, n_last, t_last = _fit_join(origin, case, states)
    n_tgt = _n_forecast(log_j0, alpha, n_last, t_last, t_tgt)
    joins = _hourly_joins(origin, case, states, log_j0, alpha, t_tgt)
    h_max = len(joins) - 1
    # Cumulative tracked players by hour, from the same join table: the
    # denominator the refresh-lag slope below is measured against.
    n_hourly = [0.0] * (h_max + 1)
    running = 0.0
    for h in range(h_max + 1):
        running += joins[h]
        n_hourly[h] = running
    t_first = _hours_since(case, origin, states[0]["at"])
    state_idx = [min(max(int(round(_hours_since(case, origin, s["at"]))), 0), h_max)
                 for s in states]

    # Observed earner increments per trophy, relative to the first read.
    first_e = {}
    increments = {}
    usable = {}
    for i, trophy in enumerate(case["trophies"]):
        base = None
        inc = []
        ok = True
        for s in states:
            rate = s["rates"][i]
            if rate is None:
                ok = False
                break
            earners = rate * s["players_tracked"] / 100.0
            if base is None:
                base = earners
            inc.append(earners - base)
        usable[trophy["id"]] = ok and len(inc) >= 2
        first_e[trophy["id"]] = base if base is not None else 0.0
        increments[trophy["id"]] = inc

    # Pooled Weibull shape: the grid value with the lowest total fit error.
    def fit_for_shape(k):
        best = {}
        total_sse = 0.0
        for lam in LAM_GRID:
            curve = _unit_curve(joins, t_first, h_max, lam, k)
            for trophy in case["trophies"]:
                tid = trophy["id"]
                if not usable[tid]:
                    continue
                inc = increments[tid]
                num = sum(d * curve[idx] for d, idx in zip(inc, state_idx)
                          if curve[idx] > 0)
                den = sum(curve[idx] ** 2 for idx in state_idx if curve[idx] > 0)
                if den <= 0:
                    continue
                p = num / den
                sse = sum(((d - p * curve[idx]) ** 2) / (abs(d) + 1.0)
                          for d, idx in zip(inc, state_idx))
                if tid not in best or sse < best[tid][0]:
                    best[tid] = (sse, lam, p)
        for tid, (sse, _lam, _p) in best.items():
            total_sse += sse
        return total_sse, best

    shape_results = {k: fit_for_shape(k) for k in K_GRID}
    k_best = min(K_GRID, key=lambda k: shape_results[k][0])
    _total, best = shape_results[k_best]

    # Shrink each trophy's log lambda toward the median of trophies whose
    # last rate sits in the same band; the weight is its earner count.
    lam_of = {tid: v[1] for tid, v in best.items()}
    bands = {}
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i] or 0.0
        band = 0 if rate < 0.1 else (1 if rate < 0.5 else
                                     (2 if rate < 2 else (3 if rate < 10 else 4)))
        bands.setdefault(band, []).append(trophy["id"])
    for ids in bands.values():
        logged = [math.log(lam_of[i]) for i in ids if i in lam_of]
        if not logged:
            continue
        med = median(logged)
        for tid in ids:
            if tid not in lam_of:
                continue
            idx = [t["id"] for t in case["trophies"]].index(tid)
            count = (last["rates"][idx] or 0) * n_last / 100.0
            w = min(1.0, count / 8.0)
            lam_of[tid] = math.exp(w * math.log(lam_of[tid]) + (1 - w) * med)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        tid = trophy["id"]
        if tid not in lam_of or not usable[tid]:
            rate = last["rates"][i]
            med = rate if rate is not None else 0.0
            sigma = math.sqrt(SIGMA_BASE ** 2 + SIGMA_N ** 2)
            rows.append({"id": tid, "q": _quantiles(med, sigma, levels)})
            continue
        curve = _unit_curve(joins, t_first, h_max, lam_of[tid], k_best)
        inc = increments[tid]
        num = sum(d * curve[idx] for d, idx in zip(inc, state_idx)
                  if curve[idx] > 0)
        den = sum(curve[idx] ** 2 for idx in state_idx if curve[idx] > 0)
        p = min(0.999, num / den) if den > 0 else 0.0
        p = max(p, 0.0)
        earners_tgt = first_e[tid] + p * curve[h_max]
        med = 100.0 * earners_tgt / max(n_tgt, 1.0)
        # Fit residual in logit space at the reads.
        residuals = []
        for j, idx in enumerate(state_idx):
            n_here = states[j]["players_tracked"]
            r_obs = states[j]["rates"][i]
            if n_here > 0 and r_obs is not None and r_obs > 0:
                r_pred = 100.0 * max(first_e[tid] + p * curve[idx], 0.0) / n_here
                residuals.append(logit(r_obs) - logit(r_pred))
        resid = math.sqrt(sum(x * x for x in residuals) / len(residuals)) \
            if residuals else 0.3
        resid = min(max(resid, 0.05), 0.8)
        # Refresh lag: the answer is the page as of Exophase's last refresh,
        # so fast-moving trophies earn width from their recent logit slope,
        # measured against the hourly denominator three hours earlier.
        h_recent = max(h_max - 3, 0)
        lg_recent = logit(100.0 * max(first_e[tid] + p * curve[h_recent], 0.0)
                          / max(n_hourly[h_recent], 1.0))
        slope = abs(logit(med) - lg_recent) / max(h_max - h_recent, 1)
        sigma_refresh = min(0.30, slope * 3.0)
        sigma = math.sqrt(SIGMA_BASE ** 2 + 0.25 * resid ** 2
                          + SIGMA_N ** 2 + sigma_refresh ** 2)
        rows.append({"id": tid, "q": _quantiles(med, sigma, levels)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
