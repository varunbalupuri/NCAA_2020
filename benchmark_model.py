import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool, cv

YEAR = 2019

def create_train_test_dfs(results, kenpoms):
    mean_kenpoms = kenpoms[['AdjEM', 'AdjO', 'AdjD', 'AdjT', 'Luck', 'OppO', 'OppD']].mean()
    l = []
    errors = []
    for _, row in results.iterrows():
        # randomly pick team1 and team2
        teams = [row['WTeamID'], row['LTeamID']]
        team1 = np.random.choice(teams)
        team2 = [i for i in teams if i != team1][0]
        winner = row['WTeamID']

        if winner == team1:
            team1wins = True
            team1score = row['WScore']
            team2score = row['LScore']
            if row['WLoc'] == 'H':
                Team1Home = True
                Team2Home = False
            elif row['WLoc'] == 'A':
                Team1Home = False
                Team2Home = True
            else:
                Team1Home = False
                Team2Home = False

        if winner == team2:
            team1wins = False
            team2score = row['WScore']
            team1score = row['LScore']
            if row['WLoc'] == 'H':
                Team1Home = False
                Team2Home = True
            elif row['WLoc'] == 'A':
                Team1Home = True
                Team2Home = False
            else:
                Team1Home = False
                Team2Home = False

        team1_kenpoms = kenpoms[kenpoms['TeamID'] == int(team1)]
        if len(team1_kenpoms) == 0:
            errors.append(team1)
            team1_kenpoms = mean_kenpoms
        else:
            team1_kenpoms = team1_kenpoms.iloc[0]

        team2_kenpoms = kenpoms[kenpoms['TeamID'] == int(team2)]
        if len(team2_kenpoms) == 0:
            errors.append(team2)
            team2_kenpoms = mean_kenpoms
        else:
            team2_kenpoms = team2_kenpoms.iloc[0]

        Diff_AdjEM = team1_kenpoms['AdjEM'] - team2_kenpoms['AdjEM']
        Diff_AdjO = team1_kenpoms['AdjO'] - team2_kenpoms['AdjO']
        Diff_AdjD = team1_kenpoms['AdjD'] - team2_kenpoms['AdjD']
        Diff_AdjT = team1_kenpoms['AdjT'] - team2_kenpoms['AdjT']
        #Diff_Luck = team1_kenpoms['Luck'] - team2_kenpoms['Luck']
        Diff_OppO = team1_kenpoms['OppO'] - team2_kenpoms['OppO']
        Diff_OppD = team1_kenpoms['OppD'] - team2_kenpoms['OppD']

        row = {
            # 'Team1ID': team1,
            # 'Team2ID': team2,
            'Team1Home': Team1Home,
            'Team2Home': Team2Home,
            # 'Team1Score': team1score,   #obviously cannot use score to predict game!
            # 'Team2Score': team2score,
            'Team1Wins': team1wins,

            'Diff_AdjEM': Diff_AdjEM,
            'Diff_AdjO': Diff_AdjO,
            'Diff_AdjD': Diff_AdjD,
            'Diff_AdjT': Diff_AdjT,
            #'Diff_Luck': Diff_Luck,
            'Diff_OppO': Diff_OppO,
            'Diff_OppD': Diff_OppD
        }

        l.append(row)
    errors = set(errors)
    print(f'error rows = {len(errors)}: {errors}')
    df = pd.DataFrame.from_records(l)

    X = df[[i for i in df.columns if i != 'Team1Wins']]
    y = df['Team1Wins'].apply(lambda x: int(x))

    return X, y


def train(year=2019):
    NCAA_tournament_results = pd.read_csv('data/mens/MNCAATourneyCompactResults.csv')
    NCAA_tournament_results = NCAA_tournament_results[NCAA_tournament_results['Season'] == year]

    NCAA_season_results = pd.read_csv('data/mens/MRegularSeasonCompactResults.csv')
    NCAA_season_results = NCAA_season_results[NCAA_season_results['Season'] == year]

    kenpoms = pd.read_csv(f'data/mens/kenpoms_{year}.csv')
    kenpoms['TeamID'] = kenpoms['TeamID'].apply(lambda x: int(x))

    X_train, y_train = create_train_test_dfs(NCAA_season_results, kenpoms)
    X_test, y_test = create_train_test_dfs(NCAA_tournament_results, kenpoms)

    categorical_idxs = [0, 1]
    model = CatBoostClassifier(custom_loss=['Logloss'], logging_level='Verbose')
    model.fit(X_train, y_train, cat_features=categorical_idxs, eval_set=(X_test, y_test), plot=False)

    feature_importances = pd.Series(dict(zip(X_train.columns,
                                             model.get_feature_importance()))).sort_values(ascending=False)
    scores = model.get_best_score()

    with open('benchmark_models_results.log', 'a') as f:
        f.write(f'------ {year} ------ ')
        f.write(str(feature_importances))
        f.write(str(scores))
        f.write('\n')

    return model

for year in range(2009,2020):
    train(year)