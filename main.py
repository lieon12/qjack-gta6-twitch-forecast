# gta6-twitch-ccu-forecast-01 entry: comparable-shape forecaster.
#
# Method (stdlib only, deterministic, no network):
#   A launch day's Twitch curve is mostly a shape, not a level: hour 1's
#   print anchors the level, and the 24-hour shape (rise after each
#   timezone unlocks, evening peak, overnight decay) is learned from the
#   42 past launches shipped in the case's `comparables`.
#   For each comparable we take its release day's hourly counts, divide by
#   its own hour-1 print and work in log space: r_h = ln(v_h / v_1).
#   The target's hour h is forecast as p1 * exp(mean_h + z_tau * sd_h),
#   where mean_h / sd_h are the kernel-weighted mean / standard deviation
#   of the comparables' r_h at that hour.  Weights are a Gaussian kernel
#   on log(hour-1 print) (bandwidth 1.0 in log units): launches whose
#   first hour looked like this one's shape the forecast most, because a
#   small hour-1 print means most of the world has yet to unlock (huge
#   growth ahead) while a big one means the launch is already global
#   (flat day).  Comparables whose platform type matches the target's
#   (console-only, rolling midnight local unlock) get a x5 weight:
#   simultaneous PC launches unlock everywhere at once and shape
#   differently.  The target is console-only (GTA VI: PS5 / Xbox Series;
#   the practice launches are too), so that match is fixed.
#   The spread comes from the comparables' own disagreement at each hour
#   (wider where launches disagree, narrower where they agree), which is
#   exactly what the pinball loss on the log scale rewards.
#   Fallback: if no comparable has a complete release day, use the house
#   baseline's persistence formula.
#
# Usage: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

import json
import math
import os
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
BW = 1.0          # Gaussian kernel bandwidth on log(hour-1 print)
PLAT_BOOST = 5.0  # weight multiplier for same platform type
SD_MULT = 0.9     # spread calibration: the raw comparable dispersion is
                  # slightly over-wide (90% coverage 0.967 on practice);
                  # 0.9 brings it to 0.924, near the honest 0.90, and also
                  # wins the leave-one-out test on the comparables.
BASE_SIGMA = 0.15  # house baseline spread, used only by the fallback


def _parse_day(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _is_console_only(platforms):
    p = (platforms or "").lower()
    has_console = ("ps5" in p or "ps4" in p or "xbox" in p or "switch" in p)
    return has_console and "pc" not in p


def _release_day_hours(comp):
    """The comparable's release-day hourly viewers, hours 1..24, or None."""
    try:
        day = _parse_day(comp["day_start"])
    except (KeyError, ValueError):
        return None
    out = {}
    for pt in comp.get("hourly") or []:
        try:
            t = _parse_day(pt["from"])
            v = pt["viewers"]
        except (KeyError, ValueError, TypeError):
            continue
        dh = (t - day).total_seconds() / 3600.0
        if 0 <= dh < 24 and dh == int(dh) and isinstance(v, (int, float)) and v > 0:
            out[int(dh) + 1] = float(v)
    if len(out) != 24:
        return None
    return [out[h] for h in range(1, 25)]


def _forecast_case(case):
    known = {k["hour"]: k for k in case.get("known") or []}
    first = known.get(1)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    # Build the comparable shape library: (log h1, console-only?, log-ratio curve).
    lib = []
    for comp in case.get("comparables") or []:
        hrs = _release_day_hours(comp)
        if hrs is None:
            continue
        h1 = hrs[0]
        lib.append(
            (
                math.log(h1),
                _is_console_only(comp.get("platforms")),
                [math.log(v / h1) for v in hrs],
            )
        )

    target_logh1 = math.log(p1)
    target_console = True  # GTA VI (and every practice launch) is console-only.
    hours = []
    for h in range(2, 25):
        if lib:
            pairs = []
            for logh1, console, curve in lib:
                w = math.exp(-0.5 * ((logh1 - target_logh1) / BW) ** 2)
                if console == target_console:
                    w *= PLAT_BOOST
                pairs.append((curve[h - 1], w))
            tot = sum(w for _, w in pairs)
            mean = sum(v * w for v, w in pairs) / tot
            var = sum(w * (v - mean) ** 2 for v, w in pairs) / tot
            sd = math.sqrt(var) * SD_MULT
            qs = [p1 * math.exp(mean + z * sd) for z in Z]
        else:
            qs = [p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1)) for z in Z]
        qs = [min(max(q, 1.0), 1e9) for q in qs]
        qs = [round(q, 2) for q in qs]
        # Keep quantiles non-decreasing after rounding.
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in inputs:
        case_id = name[: -len(".in")]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out = _forecast_case(case)
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
