"""
Compute Continual Learning metrics for Experiment 4 — multiple permutations.

Scores are normalised to [0, 1] via raw / 3 before computing CL metrics,
so that all derived metrics have a well-defined, comparable range:
  ACC  ∈ [0, 1]
  BWT  ∈ [-1, +1]
  FWT  ∈ [-1, +1]
  FM   ∈ [-1,  1]

FWT definition (Lopez-Paz & Ranzato 2017, adapted for pretrained LLMs)
───────────────────────────────────────────────────────────────────────
  FWT = (1/(T-1)) · Σ_{i=2}^{T} (A_{i-1,i} − A^base_i)
  where A_{i-1,i} = zero-shot performance on the i-th task (in training order)
                    evaluated right BEFORE training on that task
                    (above-diagonal entry of the performance matrix)
        A^base_i  = base model (SFT+DPO) performance on task i from step 0
                    (judged-generate-dpo-llama3.2-3B-full-gemma.csv)

Permutations
────────────
P1: domain_expert(A) → data_engineer(B) → rigorous_skeptic(C)
        → efficient_compute(D) → sota_chaser(E)
P2: sota_chaser(E) → efficient_compute(D) → rigorous_skeptic(C)
        → data_engineer(B) → domain_expert(A)

CL metrics (computed on normalised scores)
──────────────────────────────────────────
  ACC = (1/T)   · Σ_i  A[i, T]
  BWT = (1/T-1) · Σ_{i ≠ last} (A[i,T] − A[i, step(i)])
  FWT = (1/T-1) · Σ_{i=2}^{T} (A_{i-1,i} − A^base_i)      [NaN → skip]
          A_{i-1,i} = M[pi_i, i-1]  (above-diagonal: zero-shot before task i)
          A^base_i  = M[pi_i, 0]    (DPO baseline at step 0)
  FM  = (1/T-1) · Σ_{i ≠ last}  max_{l=1..T-1}(A[i,l]) − A[i,T]
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ─────────────────────────────────────────────────────────────────────
EXP_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "hypotheses", "final_3", "experiment_4",
)
OUT_DIR = EXP_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# ── persona catalogue (fixed row order in every matrix) ──────────────────────
PERSONA_ORDER  = ["domain_expert", "data_engineer", "rigorous_skeptic",
                   "efficient_compute", "sota_chaser"]
PERSONA_LABELS = ["A", "B", "C", "D", "E"]
T = 5  # number of tasks

# Raw score range used for normalisation  (judge scale 0–3)
SCORE_MIN, SCORE_MAX = 0.0, 3.0

METRICS = {
    "Persona Adherence": "trained_persona_adherence_score",
    "Groundedness":      "trained_groundedness_score",
    "Relevancy":         "trained_relevancy_score",
}

# Polish display names for plot titles / axis labels
METRIC_NAMES_PL = {
    "Persona Adherence": "Zgodność z personą",
    "Groundedness":      "Ugruntowanie",
    "Relevancy":         "Trafność",
}

# ── permutation definitions ───────────────────────────────────────────────────
# Each entry: label, subdir, training_order (list of persona indices 0-4),
#             step_files {step: filename}  (step 0 = DPO reference)
PERMUTATIONS = [
    {
        "label": "P1",
        "caption": "P1: A→B→C→D→E",
        "subdir": "perm1",
        # persona index trained at each step 1..T
        "training_order": [0, 1, 2, 3, 4],   # A,B,C,D,E
        "step_files": {
            0: "judged-generate-dpo-llama3.2-3B-full-gemma.csv",
            1: "judged-generate-cl-domain-llama3.2-3B-full-gemma.csv",
            2: "judged-generate-cl-A-B-llama3.2-3B-full-gemma.csv",
            3: "judged-generate-cl-A-C-llama3.2-3B-full-gemma.csv",
            4: "judged-generate-cl-A-D-llama3.2-3B-full-gemma.csv",
            5: "judged-generate-cl-A-E-llama3.2-3B-full-gemma.csv",
        },
    },
    {
        "label": "P2",
        "caption": "P2: E→D→C→B→A",
        "subdir": "perm2",
        # persona index trained at each step 1..T
        # E=sota_chaser(4), D=efficient_compute(3), C=rigorous_skeptic(2),
        # B=data_engineer(1), A=domain_expert(0)
        "training_order": [4, 3, 2, 1, 0],
        "step_files": {
            0: "judged-generate-dpo-llama3.2-3B-full-gemma.csv",
            1: "judged-generate-cl-sota-llama3.2-3B-full-gemma.csv",   # after E
            2: "judged-generate-cl-E-D-llama3.2-3B-full-gemma.csv",   # after E,D
            3: "judged-generate-cl-E-C-llama3.2-3B-full-gemma.csv",   # after E,D,C
            4: "judged-generate-cl-E-B-llama3.2-3B-full-gemma.csv",   # after E,D,C,B
            5: "judged-generate-cl-E-A-llama3.2-3B-full-gemma.csv",   # after all
        },
    },
    {
        "label": "P3",
        "caption": "P3: C→E→A→B→D",
        "subdir": "perm3",
        # C=rigorous_skeptic(2), E=sota_chaser(4), A=domain_expert(0),
        # B=data_engineer(1), D=efficient_compute(3)
        "training_order": [2, 4, 0, 1, 3],
        "step_files": {
            0: "judged-generate-dpo-llama3.2-3B-full-gemma.csv",
            1: "judged-generate-cl-rigorous-llama3.2-3B-full-gemma.csv",    # after C
            2: "judged-generate-cl-C-E-llama3.2-3B-full-gemma.csv",         # after C,E
            3: "judged-generate-cl-C-E-A-llama3.2-3B-full-gemma.csv",       # after C,E,A
            4: "judged-generate-cl-C-E-A-B-llama3.2-3B-full-gemma.csv",     # after C,E,A,B
            5: "judged-generate-cl-C-E-A-B-D-llama3.2-3B-full-gemma.csv",   # after all
        },
    },
]

# ── P2 replay variant ────────────────────────────────────────────────────────
# Same training order as P2 but steps 2-5 use replay-buffer checkpoints.
P2_REPLAY = {
    "label":          "P2-replay",
    "caption":        "P2: E→D→C→B→A (replay)",
    "subdir":         "perm2",
    "training_order": [4, 3, 2, 1, 0],
    "step_files": {
        0: "judged-generate-dpo-llama3.2-3B-full-gemma.csv",
        1: "judged-generate-cl-sota-llama3.2-3B-full-gemma.csv",   # step 1 identical
        2: "judged-replay-perm2-step2-gemma.csv",
        3: "judged-replay-perm2-step3-gemma.csv",
        4: "judged-replay-perm2-step4-gemma.csv",
        5: "judged-replay-perm2-step5-gemma.csv",
    },
}


# ── normalisation ─────────────────────────────────────────────────────────────
def normalise(x: float) -> float:
    return (x - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)


# ── load data ─────────────────────────────────────────────────────────────────
def load_means(perm: dict, step: int) -> pd.DataFrame:
    """Return per-persona mean (raw) scores for a given permutation and step."""
    path = os.path.join(EXP_DIR, perm["subdir"], perm["step_files"][step])
    df   = pd.read_csv(path)
    cols = ["user_id"] + list(METRICS.values())
    cols = [c for c in cols if c in df.columns]
    return df[cols].groupby("user_id").mean()


# ── build raw performance matrix + raw per-question data ────────────────────
def build_raw_data(perm: dict, metric_col: str) -> dict:
    """
    Return raw[i][t] = numpy array of per-question scores for persona i at step t.
    Used by bootstrap_cl_metrics.
    """
    raw: dict = {}
    for t in range(T + 1):
        path = os.path.join(EXP_DIR, perm["subdir"], perm["step_files"][t])
        df   = pd.read_csv(path)
        for i, persona in enumerate(PERSONA_ORDER):
            rows = df[df["user_id"] == persona][metric_col].dropna().values
            raw.setdefault(i, {})[t] = rows
    return raw


def build_matrix(perm: dict, metric_col: str) -> np.ndarray:
    """
    M[i, t]  = raw mean score for PERSONA_ORDER[i] at step t  (t = 0..T).
    Values in [1, 3] (original LLM-judge scale).
    NaN when the step file does not contain that persona's rows.
    """
    M = np.full((T, T + 1), np.nan)
    for t in range(T + 1):
        means = load_means(perm, t)
        for i, persona in enumerate(PERSONA_ORDER):
            if persona in means.index and metric_col in means.columns:
                M[i, t] = float(means.loc[persona, metric_col])
    return M


# ── helpers for permutation-aware step lookup ─────────────────────────────────
def step_of_persona(perm: dict) -> dict:
    """Return {persona_index: training_step} for this permutation."""
    return {pi: s + 1 for s, pi in enumerate(perm["training_order"])}


# ── CL metric functions ────────────────────────────────────────────────────────
def compute_acc(M: np.ndarray) -> float:
    """Mean final-step normalised performance across all personas."""
    return float(np.nanmean(normalise(M[:, T])))


def compute_bwt(M: np.ndarray, perm: dict) -> float:
    """
    BWT = mean over all personas except the last trained:
          A[i, T] − A[i, step(i)]
    """
    sop      = step_of_persona(perm)
    last_idx = perm["training_order"][-1]
    vals = [
        normalise(M[i, T]) - normalise(M[i, sop[i]])
        for i in range(T)
        if i != last_idx and not np.isnan(M[i, T]) and not np.isnan(M[i, sop[i]])
    ]
    return float(np.mean(vals)) if vals else float("nan")


def compute_fwt(M: np.ndarray, perm: dict) -> float:
    """
    FWT (Lopez-Paz & Ranzato 2017, adapted for pretrained LLMs):

      FWT = (1/(T-1)) · Σ_{i=2}^{T} (A_{i-1,i} − A^base_i)

    A_{i-1,i} = M[pi_i, i-1]  — zero-shot performance on the i-th task in
                                  training order, evaluated at step i-1
                                  (above-diagonal entry, before training on
                                  that task).
    A^base_i  = M[pi_i, 0]    — base model (SFT+DPO) performance on task i
                                  from step 0 (DPO reference file).

    NaN entries are skipped.
    """
    order = perm["training_order"]
    vals = []
    for k in range(1, T):       # k = i-1 for i in 2..T
        pi        = order[k]    # persona index trained at step k+1
        zero_shot = M[pi, k]    # performance at step k (before training pi)
        base      = M[pi, 0]    # DPO baseline
        if np.isnan(zero_shot) or np.isnan(base):
            continue
        vals.append(normalise(zero_shot) - normalise(base))
    return float(np.mean(vals)) if vals else float("nan")


def compute_fm(M: np.ndarray, perm: dict) -> float:
    """
    FM (Forgetting Measure, Chaudhry et al. 2018):
      FM = (1/(T-1)) · Σ_{i ≠ last}  max_{l=1..T-1} A[i,l]  −  A[i,T]

    The max is taken over ALL steps 1 to T-1 (not just from the training
    step of persona i onward), anchoring to the peak accuracy ever achieved
    before the final state. Negative values indicate positive backward
    transfer. FM ∈ [-1, 1]; lower is better (less forgetting).
    """
    last_idx = perm["training_order"][-1]
    vals = []
    for i in range(T):
        if i == last_idx:
            continue
        inter = M[i, 1:T]                    # all steps 1..T-1
        inter = inter[~np.isnan(inter)]
        if len(inter) == 0 or np.isnan(M[i, T]):
            continue
        forgetting = normalise(float(np.max(inter))) - normalise(M[i, T])
        vals.append(forgetting)              # allow negative (backward transfer)
    return float(np.mean(vals)) if vals else float("nan")


# ── statistical significance ───────────────────────────────────────────────────
def bootstrap_cl_metrics(perm: dict, metric_col: str,
                          n_boot: int = 1000, seed: int = 42) -> dict:
    """
    Bootstrap 95% CI for ACC/BWT/FWT/FM for one permutation.
    Resamples (with replacement) within each (persona, step) cell.
    Returns {cl_measure: (mean, ci_lo, ci_hi)}.
    """
    rng = np.random.default_rng(seed)
    raw = build_raw_data(perm, metric_col)
    boot: dict = {"ACC": [], "BWT": [], "FWT": [], "FM": []}
    for _ in range(n_boot):
        M_b = np.full((T, T + 1), np.nan)
        for i in range(T):
            for t in range(T + 1):
                arr = raw.get(i, {}).get(t, np.array([]))
                if len(arr) > 0:
                    M_b[i, t] = rng.choice(arr, size=len(arr), replace=True).mean()
        boot["ACC"].append(compute_acc(M_b))
        boot["BWT"].append(compute_bwt(M_b, perm))
        boot["FWT"].append(compute_fwt(M_b, perm))
        boot["FM"].append(compute_fm(M_b, perm))

    def ci95(arr):
        a = np.array([x for x in arr if not np.isnan(x)])
        return (float(np.mean(a)),
                float(np.percentile(a, 2.5)),
                float(np.percentile(a, 97.5)))

    return {m: ci95(boot[m]) for m in ["ACC", "BWT", "FWT", "FM"]}


def wilcoxon_p2_vs_replay(metric_col: str) -> dict:
    """
    Mann-Whitney U test (two-sided, unpaired) comparing P2 vs P2-replay
    at the final CL step (t=T) for each persona separately.
    Returns {persona_label: p_value}.
    """
    p2_path     = os.path.join(EXP_DIR, "perm2", PERMUTATIONS[1]["step_files"][T])
    replay_path = os.path.join(EXP_DIR, "perm2", P2_REPLAY["step_files"][T])
    df_p2     = pd.read_csv(p2_path)
    df_replay = pd.read_csv(replay_path)
    result = {}
    for i, persona in enumerate(PERSONA_ORDER):
        s1 = df_p2[df_p2["user_id"]     == persona][metric_col].dropna().values
        s2 = df_replay[df_replay["user_id"] == persona][metric_col].dropna().values
        if len(s1) >= 5 and len(s2) >= 5:
            _, p = mannwhitneyu(s1, s2, alternative="two-sided")
            result[PERSONA_LABELS[i]] = float(p)
        else:
            result[PERSONA_LABELS[i]] = float("nan")
    return result


# ── pretty print helpers ───────────────────────────────────────────────────────
def fmt(x, digits=3):
    return "—" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.{digits}f}"


def print_matrix(M, label=""):
    print(f"  {'':12s}  " + "  ".join(f"t={t}" for t in range(1, T + 1)))
    for i, pl in enumerate(PERSONA_LABELS):
        row = "  ".join(fmt(M[i, t]) for t in range(1, T + 1))
        print(f"  {pl} ({PERSONA_ORDER[i][:12]:12s}): {row}")


# ── LaTeX: per-permutation performance matrix ─────────────────────────────────
def latex_matrix(M: np.ndarray, metric_name: str, perm: dict, variant: str = "M4a (sekwencyjny)") -> str:
    """
    Produces a matrix A_{t,i} where:
      rows    = evaluation step t = 1..T  (after training the t-th task)
      columns = tasks in training order i = 1..T

    This matches the BWT formula:
      BWT = 1/(T-1) * sum_{i=1}^{T-1} (A_{T,i} - A_{i,i})
    where A_{T,i} is the last row and A_{i,i} is the diagonal.
    """
    short_metric = {"Persona Adherence": "pa", "Groundedness": "gr",
                    "Relevancy": "re"}.get(metric_name, metric_name[:2].lower())
    plab  = perm["label"].lower().replace("-", "_")
    order = perm["training_order"]   # order[k] = persona index for task k+1

    # Column headers: task i in training order, with persona letter in parens
    col_headers = " & ".join(
        rf"\textbf{{$i={t}$}} ({PERSONA_LABELS[order[t-1]]})"
        for t in range(1, T + 1)
    )

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{Macierz wydajności $\mathcal{{A}}_{{t,i}}$ -- {variant}, "
        rf"{perm['caption']}, metryka: {metric_name}. "
        rf"Wiersze: krok ewaluacji $t$; kolumny: zadanie $i$ w kolejności treningu. "
        rf"Pogrubienie: $\mathcal{{A}}_{{i,i}}$ (wynik bezpośrednio po wyuczeniu zadania).}}",
        rf"\label{{tab:exp4-matrix-{plab}-{short_metric}}}",
        r"\begin{tabular}{l " + "c " * T + r"}",
        r"\toprule",
        rf"\diagbox{{\textbf{{Krok $t$}}}}{{\textbf{{Zadanie $i$}}}} & {col_headers} \\",
        r"\midrule",
    ]

    for t in range(1, T + 1):          # rows = evaluation steps
        cells = []
        for k in range(T):             # k = column index (0-based) → task k+1
            pi   = order[k]            # persona index for task k+1
            val  = M[pi, t]
            cell = fmt(val)
            if k + 1 == t:             # diagonal: A_{i,i}
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(rf"$t={t}$ & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── LaTeX: per-metric summary tables ───────────────────────────────────────────
def latex_summary_tables(all_results: dict) -> str:
    """
    Three separate tables — one per metric — each with columns:
    Konfiguracja | ACC↑ | FWT↑ | BWT↑ | FM↓
    Rows: M4a P1 / M4a P2 / M4a P3 / M4a avg / M4b / M4c
    """
    metric_names = list(METRICS.keys())
    short_label = {
        "Persona Adherence": "pa",
        "Groundedness":      "gr",
        "Relevancy":         "re",
    }

    header = (
        r"\textbf{Konfiguracja} & "
        r"\textbf{ACC}$\uparrow$ & \textbf{FWT}$\uparrow$ & "
        r"\textbf{BWT}$\uparrow$ & \textbf{FM}$\downarrow$ \\"
    )

    def row(label, res_for_metric):
        r = res_for_metric
        return (label + " & " +
                " & ".join([fmt(r["ACC"]), fmt(r["FWT"]),
                             fmt(r["BWT"]), fmt(r["FM"])]) +
                r" \\")

    def row_avg(label, res_for_metric):
        r = res_for_metric
        def cell(m):
            mu  = r[m]
            std = r.get(m + "_std", float("nan"))
            if np.isnan(mu):
                return "--"
            if np.isnan(std):
                return f"${fmt(mu)}$"
            return rf"${fmt(mu)} \pm {fmt(std)}$"
        return (label + " & " +
                " & ".join([cell("ACC"), cell("FWT"), cell("BWT"), cell("FM")]) +
                r" \\")

    blocks = []
    for mn in metric_names:
        sl = short_label[mn]
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            rf"\caption{{Wyniki Eksperymentu 4 -- M4a (sekwencyjny), metryka: {mn}.}}",
            rf"\label{{tab:exp4-cl-metrics-{sl}}}",
            r"\begin{tabular}{l c c c c}",
            r"\toprule",
            header,
            r"\midrule",
        ]
        for perm_label, res in all_results.items():
            if perm_label == "avg":
                continue
            lines.append(row(f"M4a ({perm_label})", res[mn]))
        lines.append(r"\midrule")
        lines.append(row_avg(r"M4a (avg P1+P2+P3)", all_results["avg"][mn]))
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def latex_p2_replay_comparison(p2_res: dict, p2_replay_res: dict) -> str:
    """
    Three tables (one per metric) comparing P2 standard CL vs P2 with replay.
    """
    metric_names = list(METRICS.keys())
    short_label = {
        "Persona Adherence": "pa",
        "Groundedness":      "gr",
        "Relevancy":         "re",
    }
    header = (
        r"\textbf{Konfiguracja} & "
        r"\textbf{ACC}$\uparrow$ & \textbf{FWT}$\uparrow$ & "
        r"\textbf{BWT}$\uparrow$ & \textbf{FM}$\downarrow$ \\"
    )

    def row(label, res_for_metric):
        r = res_for_metric
        return (label + " & " +
                " & ".join([fmt(r["ACC"]), fmt(r["FWT"]),
                             fmt(r["BWT"]), fmt(r["FM"])]) +
                r" \\")

    blocks = []
    for mn in metric_names:
        sl = short_label[mn]
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            rf"\caption{{Porównanie P2 i P2 z odtwarzaniem pamięci, metryka: {mn}.}}",
            rf"\label{{tab:exp4-p2-replay-{sl}}}",
            r"\begin{tabular}{l c c c c}",
            r"\toprule",
            header,
            r"\midrule",
            row("P2", p2_res[mn]),
            row("P2 (z odtwarzaniem)", p2_replay_res[mn]),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def latex_bootstrap_ci_tables(boot_all: dict) -> str:
    """
    Three tables (one per metric) with bootstrap 95% CI.
    Rows = permutations, columns = ACC / FWT / BWT / FM with [lo, hi].
    boot_all: {perm_label: {metric_name: {cl_m: (mean, lo, hi)}}}
    """
    metric_names = list(METRICS.keys())
    short_label  = {"Persona Adherence": "pa", "Groundedness": "gr", "Relevancy": "re"}
    cl_measures  = ["ACC", "FWT", "BWT", "FM"]
    arrows       = {"ACC": r"\uparrow", "FWT": r"\uparrow",
                    "BWT": r"\uparrow", "FM": r"\downarrow"}

    header = (
        r"\textbf{Konfiguracja} & " +
        " & ".join(rf"\textbf{{{m}}}$\{arrows[m]}$" for m in cl_measures) +
        r" \\"
    )

    def cell(triple):
        mean, lo, hi = triple
        if np.isnan(mean):
            return "--"
        return rf"${fmt(mean)}\;[{fmt(lo)},\,{fmt(hi)}]$"

    blocks = []
    for mn in metric_names:
        sl = short_label[mn]
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\small",
            rf"\caption{{Bootstrap 95\% CI dla metryk CL, metryka: {mn}.}}",
            rf"\label{{tab:exp4-bootstrap-{sl}}}",
            r"\begin{tabular}{l c c c c}",
            r"\toprule",
            header,
            r"\midrule",
        ]
        for pl in [p["label"] for p in PERMUTATIONS]:
            row = f"M4a ({pl})" + " & " + " & ".join(
                cell(boot_all[pl][mn][m]) for m in cl_measures
            ) + r" \\"
            lines.append(row)
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def latex_wilcoxon_table(wilcoxon_per_metric: dict) -> str:
    """
    One table: rows = metrics, columns = personas A-E.
    Cells = p-value with significance markers (* p<0.05, ** p<0.01, *** p<0.001).
    """
    def sigmark(p):
        if np.isnan(p):
            return "--"
        stars = "^{***}" if p < 0.001 else "^{**}" if p < 0.01 else "^{*}" if p < 0.05 else ""
        return rf"${p:.3f}{stars}$"

    header = (
        r"\textbf{Metryka} & " +
        " & ".join(rf"\textbf{{{pl}}}" for pl in PERSONA_LABELS) +
        r" \\"
    )
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{Test Manna-Whitneya (dwustronny): P2 vs P2 z odtwarzaniem pamięci, $t=T$. "
        r"$^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$.}",
        r"\label{tab:exp4-wilcoxon-p2-replay}",
        r"\begin{tabular}{l c c c c c}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for mn, pvals in wilcoxon_per_metric.items():
        row = mn + " & " + " & ".join(sigmark(pvals[pl]) for pl in PERSONA_LABELS) + r" \\"
        lines.append(row)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── plots ─────────────────────────────────────────────────────────────────────
def plot_heatmap(M: np.ndarray, metric_name: str, perm: dict, out_path: str) -> None:
    sop   = step_of_persona(perm)
    order = perm["training_order"]
    data  = M[:, 1:]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=3)

    col_labels = [f"t={t} ({PERSONA_LABELS[order[t-1]]})" for t in range(1, T + 1)]
    ax.set_xticks(range(T)); ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(T))
    ax.set_yticklabels(
        [f"{PERSONA_LABELS[i]} ({PERSONA_ORDER[i].replace('_',' ')})"
         for i in range(T)], fontsize=8)
    ax.set_xlabel("Krok uczenia", fontsize=10)
    ax.set_ylabel("Oceniana persona", fontsize=10)
    ax.set_title(f"[{perm['label']}] Macierz wyników — {metric_name}",
                 fontsize=10, fontweight="bold")

    for i in range(T):
        for j in range(T):
            val = data[i, j]
            if not np.isnan(val):
                weight = "bold" if sop[i] == j + 1 else "normal"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7.5, fontweight=weight)

    # mark diagonal (training step per persona)
    for i in range(T):
        j = sop[i] - 1
        ax.add_patch(mpatches.Rectangle(
            (j - 0.5, i - 0.5), 1, 1,
            linewidth=2, edgecolor="blue", facecolor="none"))

    plt.colorbar(im, ax=ax, shrink=0.8, label="Wynik [0,3]")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_forgetting_curves(matrices_per_perm: list, out_path: str) -> None:
    """
    One row per permutation, one column per metric.
    """
    n_perms   = len(matrices_per_perm)
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(n_perms, n_metrics,
                              figsize=(5 * n_metrics, 4 * n_perms),
                              sharey=False, squeeze=False)
    steps   = list(range(1, T + 1))
    colours = plt.cm.tab10(np.linspace(0, 0.5, T))

    for row_idx, (perm, matrices) in enumerate(matrices_per_perm):
        sop   = step_of_persona(perm)
        order = perm["training_order"]
        col_labels = [f"t={t}\n({PERSONA_LABELS[order[t-1]]})" for t in steps]

        for col_idx, (metric_name, M) in enumerate(matrices.items()):
            ax = axes[row_idx][col_idx]
            for i, pl in enumerate(PERSONA_LABELS):
                persona_name = PERSONA_ORDER[i].replace("_", " ")
                y = M[i, 1:]
                ax.plot(steps, y, marker="o", color=colours[i], linewidth=1.8,
                        label=f"{pl} ({persona_name})")
                ax.axvline(x=sop[i], color=colours[i], linestyle=":", alpha=0.35)

            metric_pl = metric_name
            ax.set_title(f"[{perm['label']}] {metric_pl}", fontsize=9, fontweight="bold")
            ax.set_xlabel("Krok uczenia", fontsize=8)
            ax.set_ylabel("Wynik (skala 0–3)", fontsize=8)
            ax.set_xticks(steps)
            ax.set_xticklabels(col_labels, fontsize=7)
            ax.legend(fontsize=6, loc="lower left")
            ax.set_ylim(0, 3)
            ax.grid(True, alpha=0.3)

    plt.suptitle("Wyniki persony na przestrzeni kroków CL — Eksp. 4",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_cl_metrics_bar(all_results: dict, out_path: str) -> None:
    """Grouped bar: for each metric, show ACC/FWT/BWT/FM for P1, P2, P3, avg."""
    metric_names  = list(METRICS.keys())
    cl_names      = ["ACC", "FWT", "BWT", "FM"]
    perm_labels   = [k for k in all_results if k != "avg"] + ["avg"]
    colours_perm  = {"P1": "steelblue", "P2": "darkorange", "P3": "mediumpurple", "avg": "seagreen"}
    hatches       = {"P1": "", "P2": "//", "P3": "..", "avg": "xx"}

    n_cl   = len(cl_names)
    n_perm = len(perm_labels)
    fig, axes = plt.subplots(1, n_cl, figsize=(4 * n_cl, 5), sharey=False)

    for ax, cl_m in zip(axes, cl_names):
        x     = np.arange(len(metric_names))
        width = 0.25
        for k, pl in enumerate(perm_labels):
            vals   = [all_results[pl][mn][cl_m] for mn in metric_names]
            offset = (k - (n_perm - 1) / 2) * width
            bars   = ax.bar(x + offset, vals, width,
                            label=pl,
                            color=colours_perm[pl],
                            hatch=hatches[pl],
                            alpha=0.85)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_title(cl_m, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=8, rotation=10, ha="right")
        ax.set_ylabel("Znormalizowany wynik", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Metryki CL dla permutacji — Eksp. 4 M4a", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print("=== Experiment 4: Continual Learning Metrics (normalised) ===\n")

    # results[perm_label][metric_name] = {ACC, BWT, FWT, FM}
    all_results: dict = {}
    # matrices_per_perm: list of (perm, {metric_name: M})
    matrices_per_perm: list = []

    for perm in PERMUTATIONS:
        pl = perm["label"]
        print(f"══ Permutation {pl}: {perm['caption']} ══")
        matrices = {}
        results  = {}

        for metric_name, metric_col in METRICS.items():
            M = build_matrix(perm, metric_col)
            matrices[metric_name] = M

            acc = compute_acc(M)
            bwt = compute_bwt(M, perm)
            fwt = compute_fwt(M, perm)
            fm  = compute_fm(M,  perm)
            results[metric_name] = {"ACC": acc, "BWT": bwt, "FWT": fwt, "FM": fm}

            print(f"\n  [{metric_name}]")
            print(f"    ACC={acc:.4f}  BWT={bwt:.4f}  FWT={fmt(fwt)}  FM={fm:.4f}")
            print_matrix(M)

        all_results[pl]  = results
        matrices_per_perm.append((perm, matrices))
        print()

    # ── averaged metrics ──────────────────────────────────────────────────────
    print(f"\n══ Average P1+P2+P3 ══")
    avg_results: dict = {}
    for metric_name in METRICS:
        avg_results[metric_name] = {}
        for cl_m in ["ACC", "BWT", "FWT", "FM"]:
            vals = [all_results[p["label"]][metric_name][cl_m]
                    for p in PERMUTATIONS]
            vals = [v for v in vals if not np.isnan(v)]
            avg_results[metric_name][cl_m]          = float(np.mean(vals)) if vals else float("nan")
            avg_results[metric_name][cl_m + "_std"] = float(np.std(vals, ddof=0)) if len(vals) > 1 else float("nan")
        r = avg_results[metric_name]
        print(f"  [{metric_name}]  ACC={r['ACC']:.4f}  BWT={r['BWT']:.4f}"
              f"  FWT={fmt(r['FWT'])}  FM={r['FM']:.4f}")
    all_results["avg"] = avg_results
    print()

    # ── P2 with replay buffer ─────────────────────────────────────────────────
    print("\n══ P2 z replay bufferem ══")
    p2_replay_results: dict = {}
    p2_replay_matrices: dict = {}
    for metric_name, metric_col in METRICS.items():
        M_replay = build_matrix(P2_REPLAY, metric_col)
        p2_replay_matrices[metric_name] = M_replay
        acc = compute_acc(M_replay)
        bwt = compute_bwt(M_replay, P2_REPLAY)
        fwt = compute_fwt(M_replay, P2_REPLAY)
        fm  = compute_fm(M_replay,  P2_REPLAY)
        p2_replay_results[metric_name] = {"ACC": acc, "BWT": bwt, "FWT": fwt, "FM": fm}
        print(f"  [{metric_name}]  ACC={acc:.4f}  BWT={bwt:.4f}  FWT={fmt(fwt)}  FM={fm:.4f}")
    print()

    # ── Bootstrap confidence intervals ────────────────────────────────────────
    print("Computing bootstrap CIs (n=1000) …")
    boot_all: dict = {}
    for perm in PERMUTATIONS:
        boot_all[perm["label"]] = {}
        for metric_name, metric_col in METRICS.items():
            boot_all[perm["label"]][metric_name] = bootstrap_cl_metrics(perm, metric_col)
            print(f"  {perm['label']} [{metric_name}]: done")
    print()

    # ── Wilcoxon P2 vs P2-replay ──────────────────────────────────────────────
    print("Computing Mann-Whitney U tests (P2 vs P2-replay) …")
    wilcoxon_per_metric: dict = {}
    for metric_name, metric_col in METRICS.items():
        wilcoxon_per_metric[metric_name] = wilcoxon_p2_vs_replay(metric_col)
        res = wilcoxon_per_metric[metric_name]
        print(f"  [{metric_name}]: " + "  ".join(f"{pl}=p{res[pl]:.3f}" for pl in PERSONA_LABELS))
    print()

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    latex_path = os.path.join(OUT_DIR, "cl_metrics_exp4.tex")
    with open(latex_path, "w") as fh:
        fh.write("% Auto-generated by compute_cl_metrics_exp4.py\n")
        fh.write("% Scores normalised to [0,1] via raw/3\n\n")

        for perm, matrices in matrices_per_perm:
            fh.write(f"% ── Permutation {perm['label']} matrices ────────────────────\n\n")
            for metric_name in METRICS:
                M = matrices[metric_name]
                fh.write(latex_matrix(M, metric_name, perm))
                fh.write("\n\n")

        fh.write("% ── P2-replay matrices ──────────────────────────────────\n\n")
        for metric_name in METRICS:
            M = p2_replay_matrices[metric_name]
            fh.write(latex_matrix(M, metric_name, P2_REPLAY, variant="M4-replay (sekwencyjny z replay bufferem)"))
            fh.write("\n\n")

        fh.write("% ── Per-metric summary tables (one per metric) ──────\n\n")
        fh.write(latex_summary_tables(all_results))
        fh.write("\n\n% ── P2 replay comparison tables ─────────────────────\n\n")
        fh.write(latex_p2_replay_comparison(all_results["P2"], p2_replay_results))
        fh.write("\n\n% ── Bootstrap CI tables ─────────────────────────────────────\n\n")
        fh.write(latex_bootstrap_ci_tables(boot_all))
        fh.write("\n\n% ── Wilcoxon P2 vs P2-replay ──────────────────────────────\n\n")
        fh.write(latex_wilcoxon_table(wilcoxon_per_metric))
        fh.write("\n")

    print(f"LaTeX saved to: {latex_path}\n")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("Generating plots...")
    for perm, matrices in matrices_per_perm:
        for metric_name, M in matrices.items():
            short = metric_name.lower().replace(" ", "_")
            plot_heatmap(M, metric_name, perm,
                         os.path.join(OUT_DIR, f"heatmap_{perm['label']}_{short}.png"))

    # heatmaps for P2 replay
    for metric_name, metric_col in METRICS.items():
        M_replay = build_matrix(P2_REPLAY, metric_col)
        short    = metric_name.lower().replace(" ", "_")
        plot_heatmap(M_replay, metric_name, P2_REPLAY,
                     os.path.join(OUT_DIR, f"heatmap_P2_replay_{short}.png"))

    plot_forgetting_curves(
        matrices_per_perm,
        os.path.join(OUT_DIR, "forgetting_curves.png"))

    plot_cl_metrics_bar(
        all_results,
        os.path.join(OUT_DIR, "cl_metrics_bar.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
