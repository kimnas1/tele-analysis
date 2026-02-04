#!/usr/bin/env python3
"""
Reasoned predictor for 5G RCA questions.

This script is intentionally an "evidence -> decision -> explanation" pipeline:
it computes per-cause evidence on the *low-throughput window* (Throughput < 600),
then selects the most likely root cause and emits a short human-readable reason.

Usage:
  python3 tools/reasoned_predict.py --questions <questions.csv> --out <predicted.csv>
  python3 tools/reasoned_predict.py --questions <questions.csv> --truth <answers.csv>
"""

from __future__ import annotations

import argparse
import csv
import io
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


THROUGHPUT_LOW_MBPS = 600.0
SPEED_MAX_KMH = 40.0
RB_MIN = 160.0
OVERSHOOT_M = 1000.0
NEIGHBOR_CLOSE_DB = 6.0
MOD30_NEIGHBOR_RSRP_MIN_DBM = -110.0
MOD30_FRAC_MIN = 0.75
HANDOVERS_MIN = 2

# C1-vs-C3 guardrails (empirically stable on samples, but still explainable):
# - if UE is beyond far-edge *and* serving RSRP is weak, prefer C1 over C3 even if
#   there is a same-site close neighbor (wrong-sector) candidate.
C3_BLOCK_IF_BEYOND_AND_RSRP_LE = -87.0

# "Confident C1" stronger evidence (used to avoid C3 overriding obvious over-downtilt).
C1_STRONG_TILT_MIN_DEG = 24.0
C1_STRONG_RSRP_MAX_DBM = -85.0
C1_STRONG_MIN_ROWS = 2

# C4 is defined as severe *non-colocated* overlap; requiring it in every low row
# avoids confusing it with occasional overlaps present in other causes.
C4_DIFF_GNB_CLOSE_FRAC_MIN = 1.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def vbw_from_scenario(raw: Any) -> float:
    s = str(raw).strip().upper()
    if not s or s == "NAN":
        return 6.0
    if s == "DEFAULT":
        return 6.0
    if s.startswith("SCENARIO_"):
        try:
            n = int(s.split("_", 1)[1])
        except Exception:
            return 6.0
        if 1 <= n <= 5:
            return 6.0
        if 6 <= n <= 11:
            return 12.0
        return 25.0
    return 6.0


def as_float(x: Any) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def as_int(x: Any) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        return int(float(x))
    except Exception:
        return None


def median(values: list[float]) -> Optional[float]:
    vals = sorted(v for v in values if v is not None and not pd.isna(v))
    if not vals:
        return None
    return float(vals[len(vals) // 2])


def find_col(columns: list[str], needle: str) -> Optional[str]:
    for c in columns:
        if needle in c:
            return c
    return None


def extract_tables(question_text: str) -> tuple[list[str], list[str]]:
    user_lines: list[str] = []
    eng_lines: list[str] = []
    mode: Optional[str] = None
    for line in str(question_text).splitlines():
        if "User plane drive test data as follows" in line:
            mode = "user"
            continue
        if "Engeneering parameters data as follows" in line:
            mode = "eng"
            continue
        if "|" not in line:
            continue
        if mode == "user":
            user_lines.append(line.strip())
        elif mode == "eng":
            eng_lines.append(line.strip())
    return user_lines, eng_lines


def read_pipe_df(lines: list[str]) -> pd.DataFrame:
    if not lines:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", na_values=["-"], engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


@dataclass
class Metrics:
    n_rows: int
    n_low: int
    speed_max: Optional[float]
    avg_rb_low: Optional[float]
    max_dist_m: float
    handovers: int
    mod30_frac: float
    same_gnb_close_frac: float
    diff_gnb_close_frac: float
    beyond_beam_rows: int
    beyond_beam_denom: int
    rsrp_med_low: Optional[float]
    tilt_med_low: Optional[float]
    tilt_max_low: Optional[float]
    c1_strong_pcis: list[int]


def compute_metrics(question_text: str) -> tuple[Optional[Metrics], str]:
    user_lines, eng_lines = extract_tables(question_text)
    user_df = read_pipe_df(user_lines)
    eng_df = read_pipe_df(eng_lines)
    if user_df.empty or eng_df.empty:
        return None, "emptyTables"

    # Locate essential columns.
    tput_col = find_col(list(user_df.columns), "DL Throughput")
    pci_col = find_col(list(user_df.columns), "Serving PCI")
    rsrp_col = find_col(list(user_df.columns), "Serving SS-RSRP")
    speed_col = find_col(list(user_df.columns), "GPS Speed")
    rb_col = find_col(list(user_df.columns), "DL RB Num")

    if not all([tput_col, pci_col, rsrp_col, speed_col, rb_col]):
        return None, "missingCols"

    # Coerce numeric.
    for c in [tput_col, pci_col, rsrp_col, speed_col, rb_col, "Latitude", "Longitude"]:
        if c in user_df.columns:
            user_df[c] = pd.to_numeric(user_df[c], errors="coerce")

    for c in ["PCI", "Longitude", "Latitude", "Mechanical Downtilt", "Digital Tilt", "Height"]:
        if c in eng_df.columns:
            eng_df[c] = pd.to_numeric(eng_df[c], errors="coerce")

    # Build engineering lookup by PCI.
    eng_map: dict[int, dict[str, Any]] = {}
    for _, er in eng_df.iterrows():
        pci = as_int(er.get("PCI"))
        if pci is None:
            continue
        mech = as_float(er.get("Mechanical Downtilt"))
        dig = as_float(er.get("Digital Tilt"))
        if mech is None or dig is None:
            continue
        if int(dig) == 255:
            dig = 6.0
        eng_map[pci] = {
            "gnb": str(er.get("gNodeB ID")).strip(),
            "lat": as_float(er.get("Latitude")),
            "lon": as_float(er.get("Longitude")),
            "height": as_float(er.get("Height")),
            "tilt": mech + dig,
            "vbw": vbw_from_scenario(er.get("Beam Scenario")),
        }

    # Problem window.
    low_df = user_df[user_df[tput_col] < THROUGHPUT_LOW_MBPS]
    if low_df.empty:
        low_df = user_df

    # Neighbor column names (Top 1..5).
    n_pci_cols = [find_col(list(user_df.columns), f"Top {i} PCI") for i in range(1, 6)]
    n_rsrp_cols = [find_col(list(user_df.columns), f"Top {i} Filtered Tx BRSRP") for i in range(1, 6)]

    # Global/summary metrics.
    speed_max = as_float(user_df[speed_col].max(skipna=True))
    avg_rb_low = as_float(low_df[rb_col].mean(skipna=True))

    # Handovers across full trace.
    pcis = [as_int(v) for v in user_df[pci_col].tolist()]
    pcis = [p for p in pcis if p is not None]
    handovers = sum(1 for i in range(1, len(pcis)) if pcis[i] != pcis[i - 1])

    # Row-wise evidence on low window.
    max_dist_m = 0.0
    mod30_conf = 0
    mod30_den = 0

    same_close = 0
    diff_close = 0
    close_den = 0

    beyond_beam = 0
    beyond_den = 0

    rsrp_vals: list[float] = []
    tilt_vals: list[float] = []
    c1_strong_counts: dict[int, int] = defaultdict(int)

    for _, row in low_df.iterrows():
        s_pci = as_int(row.get(pci_col))
        s_rsrp = as_float(row.get(rsrp_col))
        ue_lat = as_float(row.get("Latitude"))
        ue_lon = as_float(row.get("Longitude"))
        if s_pci is None or s_rsrp is None or ue_lat is None or ue_lon is None:
            continue

        rsrp_vals.append(s_rsrp)

        # C6 mod30 evidence does NOT require engineering data.
        s_mod = s_pci % 30
        row_mod30 = False
        for npc_col, nrsrp_col in zip(n_pci_cols, n_rsrp_cols):
            if not npc_col or not nrsrp_col:
                continue
            n_pci = as_int(row.get(npc_col))
            n_rsrp = as_float(row.get(nrsrp_col))
            if n_pci is None or n_rsrp is None:
                continue
            if n_pci % 30 == s_mod and n_rsrp > MOD30_NEIGHBOR_RSRP_MIN_DBM:
                row_mod30 = True
                break
        mod30_den += 1
        if row_mod30:
            mod30_conf += 1

        serv = eng_map.get(s_pci)
        if not serv or serv.get("lat") is None or serv.get("lon") is None:
            continue

        if serv.get("tilt") is not None:
            tilt_vals.append(float(serv["tilt"]))

        dist = haversine_m(ue_lat, ue_lon, serv["lat"], serv["lon"])
        max_dist_m = max(max_dist_m, dist)

        # Beam-edge check (C1).
        tilt = serv.get("tilt")
        vbw = serv.get("vbw")
        height = serv.get("height")
        if tilt is not None and vbw is not None and height is not None:
            ang = tilt - vbw / 2.0
            far_edge = float("inf") if ang <= 0 else height / math.tan(math.radians(ang))
            if far_edge != float("inf"):
                beyond_den += 1
                if dist > far_edge:
                    beyond_beam += 1
                    if (
                        vbw == 6.0
                        and tilt >= C1_STRONG_TILT_MIN_DEG
                        and s_rsrp <= C1_STRONG_RSRP_MAX_DBM
                    ):
                        c1_strong_counts[s_pci] += 1

        # C3/C4 close-neighbor evidence: any co-dominant neighbor within 6dB.
        row_same = False
        row_diff = False
        for npc_col, nrsrp_col in zip(n_pci_cols, n_rsrp_cols):
            if not npc_col or not nrsrp_col:
                continue
            n_pci = as_int(row.get(npc_col))
            n_rsrp = as_float(row.get(nrsrp_col))
            if n_pci is None or n_rsrp is None:
                continue
            if n_rsrp < s_rsrp - NEIGHBOR_CLOSE_DB:
                continue
            n_eng = eng_map.get(n_pci)
            if not n_eng:
                continue
            if n_eng["gnb"] == serv["gnb"] and n_pci != s_pci:
                row_same = True
            elif n_eng["gnb"] != serv["gnb"]:
                row_diff = True
        close_den += 1
        if row_same:
            same_close += 1
        if row_diff:
            diff_close += 1

    mod30_frac = (mod30_conf / mod30_den) if mod30_den else 0.0
    same_gnb_close_frac = (same_close / close_den) if close_den else 0.0
    diff_gnb_close_frac = (diff_close / close_den) if close_den else 0.0
    rsrp_med_low = median(rsrp_vals)
    tilt_med_low = median(tilt_vals)
    tilt_max_low = max(tilt_vals) if tilt_vals else None
    c1_strong_pcis = sorted([pci for pci, cnt in c1_strong_counts.items() if cnt >= C1_STRONG_MIN_ROWS])

    return (
        Metrics(
            n_rows=int(len(user_df)),
            n_low=int(len(low_df)),
            speed_max=speed_max,
            avg_rb_low=avg_rb_low,
            max_dist_m=float(max_dist_m),
            handovers=int(handovers),
            mod30_frac=float(mod30_frac),
            same_gnb_close_frac=float(same_gnb_close_frac),
            diff_gnb_close_frac=float(diff_gnb_close_frac),
            beyond_beam_rows=int(beyond_beam),
            beyond_beam_denom=int(beyond_den),
            rsrp_med_low=rsrp_med_low,
            tilt_med_low=tilt_med_low,
            tilt_max_low=tilt_max_low,
            c1_strong_pcis=c1_strong_pcis,
        ),
        "ok",
    )


def decide(metrics: Optional[Metrics], status: str) -> tuple[str, str]:
    if metrics is None:
        return "C1", f"fallback({status})"

    m = metrics
    obs = [
        f"lowRows={m.n_low}",
        f"speedMax={int(m.speed_max) if m.speed_max is not None else 'NA'}",
        f"avgRB_low={m.avg_rb_low:.1f}" if m.avg_rb_low is not None else "avgRB_low=NA",
        f"maxDist={int(round(m.max_dist_m))}m",
        f"mod30Frac={m.mod30_frac:.2f}",
        f"handovers={m.handovers}",
        f"sameGnbCloseFrac={m.same_gnb_close_frac:.2f}",
        f"diffGnbCloseFrac={m.diff_gnb_close_frac:.2f}",
        f"beyondBeam={m.beyond_beam_rows}/{m.beyond_beam_denom}",
        f"rsrpMedLow={m.rsrp_med_low:.2f}" if m.rsrp_med_low is not None else "rsrpMedLow=NA",
        f"tiltMedLow={m.tilt_med_low:.1f}" if m.tilt_med_low is not None else "tiltMedLow=NA",
    ]

    # 1) Direct-definition causes first (they are explicitly given in the prompt).
    if m.speed_max is not None and m.speed_max > SPEED_MAX_KMH:
        return "C7", "; ".join(obs + ["decision=C7(speed>40)"])

    if m.avg_rb_low is not None and m.avg_rb_low < RB_MIN:
        return "C8", "; ".join(obs + ["decision=C8(avgRB_low<160)"])

    if m.max_dist_m > OVERSHOOT_M:
        return "C2", "; ".join(obs + ["decision=C2(maxDist>1000m)"])

    # 2) Interference / mobility.
    if m.mod30_frac >= MOD30_FRAC_MIN:
        return "C6", "; ".join(obs + ["decision=C6(mod30 persistent)"])

    if m.handovers >= HANDOVERS_MIN:
        return "C5", "; ".join(obs + ["decision=C5(frequent HOs)"])

    # 3) Neighbor geometry/selection.
    if m.diff_gnb_close_frac >= C4_DIFF_GNB_CLOSE_FRAC_MIN:
        if (
            abs(m.same_gnb_close_frac - 1.0) < 1e-9
            and m.beyond_beam_rows > 0
            and m.rsrp_med_low is not None
            and m.rsrp_med_low <= C3_BLOCK_IF_BEYOND_AND_RSRP_LE
        ):
            return "C1", "; ".join(obs + ["decision=C1(weak+beam-edge, override C4)"])
        return "C4", "; ".join(obs + ["decision=C4(non-colocated overlap in all low rows)"])

    # 4) Coverage evidence (C1) that should not be overridden by C3.
    if m.c1_strong_pcis:
        return "C1", "; ".join(obs + [f"decision=C1(confident downtilt, pci={m.c1_strong_pcis})"])

    # 5) Same-site neighbor present in every low row: likely wrong serving (C3),
    # unless it looks like a coverage-limited edge case (C1).
    if abs(m.same_gnb_close_frac - 1.0) < 1e-9:
        if (
            m.beyond_beam_denom > 0
            and m.beyond_beam_rows == m.beyond_beam_denom
            and m.beyond_beam_denom == m.n_low
        ):
            return "C1", "; ".join(obs + ["decision=C1(all low rows beyond far-edge, block C3)"])
        if m.tilt_med_low is not None and m.tilt_med_low >= 30.0 and m.beyond_beam_rows > 0:
            return "C1", "; ".join(obs + ["decision=C1(extreme downtilt, block C3)"])
        if (
            m.beyond_beam_rows > 0
            and m.rsrp_med_low is not None
            and m.rsrp_med_low <= C3_BLOCK_IF_BEYOND_AND_RSRP_LE
        ):
            return "C1", "; ".join(obs + ["decision=C1(weak+beam-edge, block C3)"])
        return "C3", "; ".join(obs + ["decision=C3(same-site close neighbor in all low rows)"])

    # 6) Default coverage explanation if beam geometry shows UE beyond the main lobe.
    if m.beyond_beam_rows > 0:
        return "C1", "; ".join(obs + ["decision=C1(UE beyond far-edge)"])

    return "C1", "; ".join(obs + ["decision=C1(fallback)"])


def load_truth(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["ID"].strip()] = row["answer"].strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True, help="CSV with columns: ID,question")
    ap.add_argument("--out", help="Write predictions CSV: ID,answer,reason")
    ap.add_argument("--truth", help="Optional ground truth CSV: ID,answer (prints accuracy)")
    args = ap.parse_args()

    truth = load_truth(args.truth) if args.truth else None
    rows_out: list[dict[str, str]] = []

    conf: dict[str, Counter] = defaultdict(Counter)
    mismatches: list[tuple[str, str, str]] = []

    with open(args.questions, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["ID"].strip()
            metrics, status = compute_metrics(row["question"])
            pred, reason = decide(metrics, status)
            rows_out.append({"ID": qid, "answer": pred, "reason": reason})

            if truth is not None and qid in truth:
                gt = truth[qid]
                conf[gt][pred] += 1
                if pred != gt:
                    mismatches.append((qid, pred, gt))

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ID", "answer", "reason"])
            w.writeheader()
            w.writerows(rows_out)

    if truth is not None:
        total = sum(sum(c.values()) for c in conf.values())
        wrong = len(mismatches)
        acc = (total - wrong) / total if total else 0.0
        print(f"Accuracy: {total-wrong}/{total} ({acc*100:.2f}%)")
        labels = [f"C{i}" for i in range(1, 9)]
        for gt_label in labels:
            if gt_label not in conf:
                continue
            parts = [f"{p}:{conf[gt_label][p]}" for p in labels if conf[gt_label][p]]
            print(f"  {gt_label} -> {', '.join(parts)}")
        if mismatches:
            print("Mismatches (first 25):")
            for qid, p, g in mismatches[:25]:
                print(f"  {qid} pred={p} gt={g}")


if __name__ == "__main__":
    main()
