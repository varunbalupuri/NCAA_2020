import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool, cv

YEAR = 2019

def generate_season_statistics(detailed_match_results, season):
    teamids = list(set(detailed_match_results['LTeamID'].unique().tolist() + detailed_match_results['WTeamID'].unique().tolist()))
    d = {}
    for teamid in teamids:
        stats = ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']

        winning_games = detailed_match_results[(detailed_match_results['WTeamID'] == teamid) &
                                               (detailed_match_results['Season'] == season)][
                        [f'W{s}' for s in stats]]
        winning_games.columns = [colname.lstrip('W') for colname in winning_games.columns]
        losing_games = detailed_match_results[(detailed_match_results['LTeamID'] == teamid) &
                                              (detailed_match_results['Season'] == season)][
                        [f'L{s}' for s in stats]]
        losing_games.columns = [colname.lstrip('L') for colname in losing_games.columns]
        d[teamid] = pd.concat([winning_games, losing_games]).mean()
    return pd.DataFrame(d).T


def last_14_day_win_rates(detailed_match_results):
    teamids = list(
        set(detailed_match_results['LTeamID'].unique().tolist() + detailed_match_results['WTeamID'].unique().tolist()))
    d = {}
    for teamid in teamids:
        winning_games = detailed_match_results[(detailed_match_results['WTeamID'] == teamid) &
                                               (detailed_match_results['DayNum'] > 118)].shape[0]
        losing_games = detailed_match_results[(detailed_match_results['LTeamID'] == teamid) &
                                               (detailed_match_results['DayNum'] > 118)].shape[0]
        try:
            wr = winning_games / (winning_games + losing_games)
        except ZeroDivisionError:
            wr = 0
        d[teamid] = wr
    return d


def create_train_test_dfs(results, kenpoms, massey_ordinals, seeds, season_stats, winrates):
    l = []
    errors = []
    mean_kenpoms = kenpoms[['AdjEM','AdjO','AdjD','AdjT','Luck','OppO','OppD']].mean()
    for _, row in results.iterrows():
        # repeat this entire thing 3 times
        for iii in [0, 1]:
            # randomly pick team1 and team2
            teams = [row['WTeamID'], row['LTeamID']]
            team1 = teams[iii]
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
            Diff_Luck = team1_kenpoms['Luck'] - team2_kenpoms['Luck']
            Diff_OppO = team1_kenpoms['OppO'] - team2_kenpoms['OppO']
            Diff_OppD = team1_kenpoms['OppD'] - team2_kenpoms['OppD']


            # box season averages
            t1_stats = season_stats.loc[team1]
            t2_stats = season_stats.loc[team2]

            diff_stats = t1_stats - t2_stats

            # massey ordinals
            try:
                team1_massey = massey_ordinals[
                    (massey_ordinals['RankingDayNum'] < row['DayNum']) &
                    (massey_ordinals['TeamID'] == team1)
                    ].groupby('RankingDayNum').mean().idxmax()['OrdinalRank']
            except Exception:
                team1_massey = \
                massey_ordinals[(massey_ordinals['TeamID'] == team1)].groupby('RankingDayNum').mean().iloc[0]['OrdinalRank']

            try:
                team2_massey = massey_ordinals[
                    (massey_ordinals['RankingDayNum'] < row['DayNum']) &
                    (massey_ordinals['TeamID'] == team2)
                    ].groupby('RankingDayNum').mean().idxmax()['OrdinalRank']

            except Exception:
                team2_massey = \
                massey_ordinals[(massey_ordinals['TeamID'] == team2)].groupby('RankingDayNum').mean().iloc[0]['OrdinalRank']

            Diff_MasseyOrdinal = team1_massey - team2_massey

            try:
                team1_seed = seeds[seeds['TeamID'] == team1].iloc[0]['seednumber']
            except:
                team1_seed = 16
            try:
                team2_seed = seeds[seeds['TeamID'] == team2].iloc[0]['seednumber']
            except:
                team2_seed = 16

            Diff_seeds = team1_seed - team2_seed

            #winrate
            t1_winrate_14_days = winrates[team1]
            t2_winrate_14_days = winrates[team2]


            new_row = {
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
                'Diff_Luck': Diff_Luck,
                'Diff_OppO': Diff_OppO,
                'Diff_OppD': Diff_OppD,
                'Team1WinRate': t1_winrate_14_days,
                'Team2WinRate': t2_winrate_14_days,
                'Diff_MasseyOrdinal': Diff_MasseyOrdinal,
                'Diff_seeds': Diff_seeds
            }
            new_row.update(diff_stats)
            l.append(new_row)
    errors = set(errors)
    print(f'error teamids = {errors}')
    df = pd.DataFrame.from_records(l)

    X = df[[i for i in df.columns if i != 'Team1Wins']]
    y = df['Team1Wins'].apply(lambda x: int(x))

    return X, y


def train(year=2019, kenpom_lag=False, all_years=False):
    NCAA_tournament_results = pd.read_csv('data/mens/MNCAATourneyDetailedResults.csv')
    NCAA_tournament_results = NCAA_tournament_results[NCAA_tournament_results['Season'] == year]

    NCAA_season_results = pd.read_csv('data/mens/MRegularSeasonDetailedResults.csv')
    NCAA_season_results = NCAA_season_results[NCAA_season_results['Season'] == year]

    season_stats = generate_season_statistics(NCAA_season_results, year)

    # kenpoms
    if kenpom_lag:
        k_year = year-1
    else:
        k_year = year
    kenpoms = pd.read_csv(f'data/mens/PiT_kenpoms_{k_year}.csv')
    kenpoms['TeamID'] = kenpoms['TeamID'].apply(lambda x: int(x))

    # massey ordinals
    massey_ordinals = pd.read_csv('data/mens/MMasseyOrdinals.csv')
    massey_ordinals = massey_ordinals[massey_ordinals['Season'] == year]

    # seeds
    seeds = pd.read_csv('data/mens/MNCAATourneySeeds.csv')
    seeds = seeds[seeds['Season'] == year]
    seeds['seednumber'] = seeds['Seed'].apply(lambda x: int(''.join([i for i in x if i.isdigit()])))
    seeds['region'] = seeds['Seed'].apply(lambda x: x[0])

    winrates = last_14_day_win_rates(NCAA_season_results)

    X_train, y_train = create_train_test_dfs(NCAA_season_results, kenpoms, massey_ordinals,
                                             seeds, season_stats, winrates)
    X_test, y_test = create_train_test_dfs(NCAA_tournament_results, kenpoms, massey_ordinals,
                                           seeds, season_stats, winrates)

    categorical_idxs = [0, 1]
    model = CatBoostClassifier(custom_loss=['Logloss'], logging_level='Verbose')
    model.fit(X_train, y_train, cat_features=categorical_idxs, eval_set=(X_test, y_test), plot=False)

    feature_importances = pd.Series(dict(zip(X_train.columns,
                                             model.get_feature_importance()))).sort_values(ascending=False)
    scores = model.get_best_score()

    with open('benchmark2_models_results.log', 'a') as f:
        f.write(f'------ {year} ------ ')
        f.write(str(feature_importances))
        f.write(str(scores))
        f.write('\n')

    return model


def train_giant_model(start_year=2011, end_year=2019):
    X_trains = []
    y_trains = []
    X_tests = []
    y_tests = []
    for year in range(start_year, end_year+1):
        print(year)
        NCAA_tournament_results = pd.read_csv('data/mens/MNCAATourneyDetailedResults.csv')
        NCAA_tournament_results = NCAA_tournament_results[NCAA_tournament_results['Season'] == year]

        NCAA_season_results = pd.read_csv('data/mens/MRegularSeasonDetailedResults.csv')
        NCAA_season_results = NCAA_season_results[NCAA_season_results['Season'] == year]

        season_stats = generate_season_statistics(NCAA_season_results, year)

        # # kenpoms
        # if kenpom_lag:
        #     k_year = year - 1
        # else:
        #     k_year = year
        kenpoms = pd.read_csv(f'data/mens/PiT_kenpoms_{year}.csv')
        kenpoms['TeamID'] = kenpoms['TeamID'].apply(lambda x: int(x))

        # massey ordinals
        massey_ordinals = pd.read_csv('data/mens/MMasseyOrdinals.csv')
        massey_ordinals = massey_ordinals[massey_ordinals['Season'] == year]

        # seeds
        seeds = pd.read_csv('data/mens/MNCAATourneySeeds.csv')
        seeds = seeds[seeds['Season'] == year]
        seeds['seednumber'] = seeds['Seed'].apply(lambda x: int(''.join([i for i in x if i.isdigit()])))
        seeds['region'] = seeds['Seed'].apply(lambda x: x[0])

        winrates = last_14_day_win_rates(NCAA_season_results)

        X_train, y_train = create_train_test_dfs(NCAA_season_results, kenpoms, massey_ordinals,
                                                 seeds, season_stats, winrates)
        X_test, y_test = create_train_test_dfs(NCAA_tournament_results, kenpoms, massey_ordinals,
                                               seeds, season_stats, winrates)

        X_trains.append(X_train)
        y_trains.append(y_train)
        X_tests.append(X_test)
        y_tests.append(y_test)

    X_train = pd.concat(X_trains)
    y_train = pd.concat(y_trains)
    X_test = pd.concat(X_tests)
    y_test = pd.concat(y_tests)

    return X_train, y_train, X_test, y_test


def simulate_oos(year=2019):
    NCAA_tournament_results = pd.read_csv('data/mens/MNCAATourneyDetailedResults.csv')
    NCAA_tournament_results = NCAA_tournament_results[NCAA_tournament_results['Season'] == year]

    NCAA_season_results = pd.read_csv('data/mens/MRegularSeasonDetailedResults.csv')
    NCAA_season_results = NCAA_season_results[NCAA_season_results['Season'] == year]

    season_stats = generate_season_statistics(NCAA_season_results, year)
    kenpoms = pd.read_csv(f'data/mens/PiT_kenpoms_{year}.csv')
    kenpoms['TeamID'] = kenpoms['TeamID'].apply(lambda x: int(x))
    massey_ordinals = pd.read_csv('data/mens/MMasseyOrdinals.csv')
    massey_ordinals = massey_ordinals[massey_ordinals['Season'] == year]

    # seeds
    seeds = pd.read_csv('data/mens/MNCAATourneySeeds.csv')
    seeds = seeds[seeds['Season'] == year]
    seeds['seednumber'] = seeds['Seed'].apply(lambda x: int(''.join([i for i in x if i.isdigit()])))
    seeds['region'] = seeds['Seed'].apply(lambda x: x[0])
    winrates = last_14_day_win_rates(NCAA_season_results)
    X, y = create_train_test_dfs(NCAA_tournament_results, kenpoms, massey_ordinals,
                                 seeds, season_stats, winrates)
    return X, y


def predict_oos():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', default=2019)
    parser.add_argument('--kenpom_lag', default=False, action='store_true')
    parser.add_argument('--all_years', default=False, action='store_true')

    args = parser.parse_args()

