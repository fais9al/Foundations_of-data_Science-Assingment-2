# World Cup 2026 Data Analysis

My name is Faisal Hossain and my student ID is 403084. This repository contains my individual analysis for HIT140 Assessment 2.

## My question

Is there a difference in the average shots on target per 90 minutes between forwards and midfielders?

## What I did

I used Python to:

- Clean and prepare the dataset
- Select players who played at least 270 minutes
- Take a random sample of 40 forwards and 40 midfielders
- Calculate descriptive statistics and confidence intervals
- Perform a Welch two-sample t-test
- Create graphs to compare the groups

## Files

- `analysis.py` contains the Python code
- `data` contains the World Cup dataset
- `outputs` contains the results and graph
- `requirements.txt` lists the Python libraries

## Running the code

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the analysis:

```bash
python analysis.py
```

## Result

The sampled forwards had a higher average number of shots on target per 90 minutes than the sampled midfielders. The statistical test found that the difference was significant.
