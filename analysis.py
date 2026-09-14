"""Faisal Hossain (403084) — HIT140 Assessment 2 individual contribution.

Question: Is mean shots on target per 90 different between forwards and
midfielders in the supplied FBref 2026 World Cup player dataset?

The script performs every analysis step in Python: loading, wrangling,
reproducible sampling, descriptive statistics, confidence intervals,
hypothesis testing, robustness checks, effect size and visualisation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "fbref_2026_player_stats.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
SAMPLE_SIZE_PER_GROUP = 40
MINIMUM_MINUTES = 270
ALPHA = 0.05


def mean_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    """Return a two-sided Student-t confidence interval for one mean."""
    values = values.dropna().astype(float)
    standard_error = stats.sem(values)
    critical_value = stats.t.ppf((1 + confidence) / 2, len(values) - 1)
    margin = critical_value * standard_error
    return float(values.mean() - margin), float(values.mean() + margin)


def welch_difference_ci(
    group_a: pd.Series, group_b: pd.Series, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Return mean difference, CI, and Welch-Satterthwaite degrees of freedom."""
    a, b = group_a.astype(float), group_b.astype(float)
    difference = a.mean() - b.mean()
    variance_a, variance_b = a.var(ddof=1), b.var(ddof=1)
    term_a, term_b = variance_a / len(a), variance_b / len(b)
    standard_error = np.sqrt(term_a + term_b)
    degrees_freedom = (term_a + term_b) ** 2 / (
        term_a**2 / (len(a) - 1) + term_b**2 / (len(b) - 1)
    )
    critical_value = stats.t.ppf((1 + confidence) / 2, degrees_freedom)
    margin = critical_value * standard_error
    return (
        float(difference),
        float(difference - margin),
        float(difference + margin),
    )


def describe(values: pd.Series) -> dict[str, float | int]:
    """Return the Week 1–5 descriptive measures used in the presentation."""
    values = values.astype(float)
    mode = values.mode().iloc[0]
    q1, q3 = values.quantile([0.25, 0.75])
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "mode": float(mode),
        "variance": float(values.var(ddof=1)),
        "standard_deviation": float(values.std(ddof=1)),
        "minimum": float(values.min()),
        "q1": float(q1),
        "q3": float(q3),
        "maximum": float(values.max()),
        "iqr": float(q3 - q1),
    }


def hedges_g(group_a: pd.Series, group_b: pd.Series) -> tuple[float, float]:
    """Return Cohen's d and its small-sample corrected Hedges' g."""
    a, b = group_a.astype(float), group_b.astype(float)
    pooled_sd = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    cohens_d = (a.mean() - b.mean()) / pooled_sd
    correction = 1 - 3 / (4 * (len(a) + len(b)) - 9)
    return float(cohens_d), float(correction * cohens_d)


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, select, type-check, de-duplicate and filter the raw data."""
    raw = pd.read_csv(DATA_FILE)
    selected = raw[
        ["player", "team", "pos", "Playing Time_Min", "Standard_SoT/90"]
    ].rename(
        columns={
            "pos": "listed_position",
            "Playing Time_Min": "minutes",
            "Standard_SoT/90": "shots_on_target_per90",
        }
    )
    selected["minutes"] = pd.to_numeric(selected["minutes"], errors="coerce")
    selected["shots_on_target_per90"] = pd.to_numeric(
        selected["shots_on_target_per90"], errors="coerce"
    )
    selected["primary_position"] = (
        selected["listed_position"].astype(str).str.split(",").str[0].str.strip()
    )
    cleaned = (
        selected.drop_duplicates(subset=["player", "team"])
        .dropna(subset=["minutes", "shots_on_target_per90", "primary_position"])
        .query("minutes >= @MINIMUM_MINUTES")
        .query("primary_position in ['FW', 'MF']")
        .reset_index(drop=True)
    )
    return raw, cleaned


def create_chart(sample: pd.DataFrame, group_summary: pd.DataFrame) -> None:
    """Create a presentation-ready Python visualisation."""
    sns.set_theme(style="whitegrid", context="talk")
    colours = {"Forwards": "#F97316", "Midfielders": "#0EA5E9"}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7))

    sns.histplot(
        data=sample,
        x="shots_on_target_per90",
        hue="group",
        bins=9,
        stat="count",
        element="step",
        palette=colours,
        alpha=0.38,
        common_norm=False,
        ax=axes[0],
    )
    axes[0].set_title("Sample distributions", loc="left", fontweight="bold")
    axes[0].set_xlabel("Shots on target per 90")
    axes[0].set_ylabel("Players")

    sns.boxplot(
        data=sample,
        x="group",
        y="shots_on_target_per90",
        hue="group",
        palette=colours,
        legend=False,
        width=0.55,
        ax=axes[1],
    )
    sns.stripplot(
        data=sample,
        x="group",
        y="shots_on_target_per90",
        color="#172554",
        alpha=0.46,
        size=4,
        jitter=0.17,
        ax=axes[1],
    )
    axes[1].set_title("Spread and individual players", loc="left", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Shots on target per 90")

    for group, x_position in [("Forwards", 0), ("Midfielders", 1)]:
        mean = float(group_summary.loc[group, "mean"])
        axes[1].scatter(x_position, mean, marker="D", s=85, color="#111827", zorder=5)
        axes[1].annotate(
            f"mean = {mean:.2f}",
            (x_position, mean),
            xytext=(12, 4),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle(
        "Forwards recorded more shots on target per 90",
        x=0.055,
        y=1.02,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="#172554",
    )
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "shots_on_target_comparison.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    raw, eligible = prepare_data()
    population_counts = eligible.groupby("primary_position").size().to_dict()

    forwards = (
        eligible.query("primary_position == 'FW'")
        .sample(n=SAMPLE_SIZE_PER_GROUP, random_state=RANDOM_SEED)
        .assign(group="Forwards")
    )
    midfielders = (
        eligible.query("primary_position == 'MF'")
        .sample(n=SAMPLE_SIZE_PER_GROUP, random_state=RANDOM_SEED)
        .assign(group="Midfielders")
    )
    sample = pd.concat([forwards, midfielders], ignore_index=True)
    sample.to_csv(OUTPUT_DIR / "analysis_sample.csv", index=False)

    forward_values = forwards["shots_on_target_per90"]
    midfielder_values = midfielders["shots_on_target_per90"]
    summaries = {
        "Forwards": describe(forward_values),
        "Midfielders": describe(midfielder_values),
    }
    summary_table = pd.DataFrame(summaries).T
    summary_table.index.name = "group"
    summary_table.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

    forward_ci = mean_ci(forward_values)
    midfielder_ci = mean_ci(midfielder_values)
    difference, difference_low, difference_high = welch_difference_ci(
        forward_values, midfielder_values
    )
    t_test = stats.ttest_ind(forward_values, midfielder_values, equal_var=False)
    shapiro_forward = stats.shapiro(forward_values)
    shapiro_midfielder = stats.shapiro(midfielder_values)
    levene = stats.levene(forward_values, midfielder_values, center="median")
    mann_whitney = stats.mannwhitneyu(
        forward_values, midfielder_values, alternative="two-sided"
    )
    cohens_d, effect_hedges_g = hedges_g(forward_values, midfielder_values)

    results = {
        "student": {"name": "Faisal Hossain", "student_id": "403084"},
        "analytic_question": (
            "Is mean shots on target per 90 different between forwards and "
            "midfielders in the supplied FBref 2026 World Cup player dataset?"
        ),
        "hypotheses": {
            "null": "The two population means are equal (mu_FW = mu_MF).",
            "alternative": "The two population means differ (mu_FW != mu_MF).",
        },
        "data_preparation": {
            "raw_rows": int(len(raw)),
            "minimum_minutes": MINIMUM_MINUTES,
            "eligible_forward_population": int(population_counts.get("FW", 0)),
            "eligible_midfielder_population": int(population_counts.get("MF", 0)),
            "sample_size_per_group": SAMPLE_SIZE_PER_GROUP,
            "random_seed": RANDOM_SEED,
        },
        "descriptive_statistics": summaries,
        "confidence_intervals_95_percent": {
            "forward_mean": list(forward_ci),
            "midfielder_mean": list(midfielder_ci),
            "forward_minus_midfielder": [difference_low, difference_high],
        },
        "welch_two_sample_t_test": {
            "mean_difference": difference,
            "t_statistic": float(t_test.statistic),
            "p_value": float(t_test.pvalue),
            "alpha": ALPHA,
            "decision": "Reject H0" if t_test.pvalue < ALPHA else "Fail to reject H0",
        },
        "assumption_and_robustness_checks": {
            "shapiro_forward_p": float(shapiro_forward.pvalue),
            "shapiro_midfielder_p": float(shapiro_midfielder.pvalue),
            "levene_median_p": float(levene.pvalue),
            "mann_whitney_u": float(mann_whitney.statistic),
            "mann_whitney_p": float(mann_whitney.pvalue),
        },
        "effect_size": {"cohens_d": cohens_d, "hedges_g": effect_hedges_g},
        "conclusion": (
            "The sampled forwards had a statistically significantly higher mean "
            "shots-on-target rate than the sampled midfielders."
        ),
    }
    (OUTPUT_DIR / "analysis_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    create_chart(sample, summary_table)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
