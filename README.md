# NCAA 2020 Kaggle Forecasting Competition Entry.

Each season there are thousands of NCAA basketball games played between Division I men's teams, culminating in March Madness®, the 68-team national championship that starts in the middle of March.

This repository contains code to gather Kenneth Pomeroy Historical Basketball Ratings (using internet archives where neccesary), build a featureset using provided data
and kenpom data, using lagged features, Massey Ordinals, Pomeroy Ordinals, Home advantage and EWMA's of winning games for a team and various other synthetically generated features
with the goal of predicting either the probability of winning directly (classification) or via the difference in terminal score (regression) between two teams
using Gradient Boosted Trees.

See `benchmark_model.py` for details

Submissions are scored on the mean of the log loss of probability of winning across all teams.

There are 68 teams, meaning (68*67)/2 = 2278 potential matchups, this code must generate a prediction for every potential matchup

Featuresets are generated completely point-in-time.

### Data

See https://www.kaggle.com/c/ncaam-march-mania-2021/data for specifications of provided data for the competition.

### Performance on Historical Data:

Four models are created (details in notebooks), the total log-loss on out-of-sample real matches in the NCAA tournament of that year
are as follows (for the classification method):

```
        normal	        aggressive	clipped	    tail_risk_fatter   super_risky
2015	0.530645	0.526424	0.532765	0.534373	0.524100
2016	0.612272	1.078864	0.613786	0.608093	1.086789
2017	0.535180	0.531384	0.537909	0.534439	0.536696
2018	0.619928	1.072146	0.608639	0.632761	1.084978
2019	0.499275	0.496065	0.502500	0.502311	0.486998
```

(anything below 0.5 is ok-ish, anything below 0.45 is very good)