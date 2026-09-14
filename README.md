# Faisal Hossain — Assessment 2 individual contribution

**Student ID:** 403084  
**Unit:** HIT140  
**Dataset:** FBref FIFA World Cup 2026 player statistics  
**Analytic question:** Is mean shots on target per 90 different between forwards and midfielders?

## What this folder contains

- `analysis.py`: complete reproducible Python analysis.
- `data/fbref_2026_player_stats.csv`: supplied 2026 player-level dataset.
- `outputs/`: the sampled observations, descriptive statistics, test results, and Python visualisation.
- `presentation/Faisal_2026_World_Cup_Analysis.pptx`: three-slide individual segment.
- `presentation/Faisal_Speaking_Script.md`: speaking script designed for less than three minutes.
- `SOURCES.md`: data provenance and references.
- `AI_USAGE_NOTES.txt`: truthful wording to adapt for the required declaration.
- `SUBMISSION_CHECKLIST.txt`: final recording and submission checks.

## Reproduce the analysis

From this folder, run:

```bash
python -m pip install -r requirements.txt
python analysis.py
```

The script uses a balanced reproducible sample of 40 eligible forwards and 40 eligible midfielders. Eligibility requires at least 270 playing minutes, and the random seed is 42.

## Statistical method

The response variable is `Standard_SoT/90`. The explanatory variable is primary playing position (`FW` or `MF`). The analysis reports descriptive statistics, 95% confidence intervals, a two-sided Welch independent-samples t-test, effect size, assumption checks, and a Mann–Whitney sensitivity check.

This package is an individual contribution only. The official assessment instructions require a group submission, so it must be coordinated with the lecturer and remaining group submission materials.
