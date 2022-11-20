"""
The data process is base on https://www.kaggle.com/code/sslp23/predicting-fifa-2022-world-cup-with-ml
"""
import os.path

import pandas as pd
from sklearn.model_selection import train_test_split

from configs.config import cfg
from configs.constants import DATA_ROOT


def result_finder(home, away):
    """
    Encode the data
    :param home:
    :param away:
    :return:
    """
    if home > away:
        return pd.Series([0, 3, 0])
    if home < away:
        return pd.Series([1, 0, 3])
    else:
        return pd.Series([2, 1, 1])


def create_dataset(df: pd.DataFrame):
    """
    Create train, test dataset
    :param df:
    :return:
    """
    x_, y = df.iloc[:, 3:], df[["target"]]
    x_train, x_test, y_train, y_test = train_test_split(
        x_, y, test_size=0.22, random_state=100)
    return x_train, x_test, y_train, y_test


def data_preparing():
    """
    Data preparing
    :return:
    """
    try:
        df = pd.read_csv(cfg.data.result_url)
    except Exception as e:
        print(e)
        df = pd.read_csv(os.path.join(DATA_ROOT, cfg.data.result_file))
    df["date"] = pd.to_datetime(df["date"])
    df.dropna(inplace=True)
    df = df[(df["date"] >= cfg.day_get_result)].reset_index(drop=True)

    # RANK data prepare
    rank = pd.read_csv(os.path.join(DATA_ROOT, cfg.data.rank_file))
    rank["rank_date"] = pd.to_datetime(rank["rank_date"])
    rank = rank[(rank["rank_date"] >= cfg.day_get_rank)].reset_index(drop=True)
    rank["country_full"] = rank["country_full"].str.replace("IR Iran", "Iran").str.replace("Korea Republic",
                                                                                           "South Korea").str.replace(
        "USA", "United States")

    # The merge is made in order to get a dataset FIFA games and its rankings.
    rank = rank.set_index(['rank_date']).groupby(['country_full'], group_keys=False).resample('D').first().fillna(
        method='ffill').reset_index()
    df_wc_ranked = df.merge(
        rank[["country_full", "total_points", "previous_points", "rank", "rank_change", "rank_date"]],
        left_on=["date", "home_team"], right_on=["rank_date", "country_full"]).drop(["rank_date", "country_full"],
                                                                                    axis=1)

    df_wc_ranked = df_wc_ranked.merge(
        rank[["country_full", "total_points", "previous_points", "rank", "rank_change", "rank_date"]],
        left_on=["date", "away_team"], right_on=["rank_date", "country_full"], suffixes=("_home", "_away")).drop(
        ["rank_date", "country_full"], axis=1)

    # Featuring
    df = df_wc_ranked

    df[["result", "home_team_points", "away_team_points"]] = df.apply(
        lambda x: result_finder(x["home_score"], x["away_score"]), axis=1)

    # we create columns that will help in the creation of the features: ranking difference,
    # points won at the game vs. team faced rank, and goals difference in the game.
    # All features that are not differences should be created for the two teams (away and home).
    df["rank_dif"] = df["rank_home"] - df["rank_away"]
    df["sg"] = df["home_score"] - df["away_score"]
    df["points_home_by_rank"] = df["home_team_points"] / df["rank_away"]
    df["points_away_by_rank"] = df["away_team_points"] / df["rank_home"]

    # In order to create the features, I'll separate the dataset in home team's and away team's dataset,
    # unify them and calculate the past game values.
    # After that, I'll separate again and merge them, retrieving the original dataset.
    # This process optimizes the creation of the features.
    home_team = df[["date", "home_team", "home_score", "away_score", "rank_home", "rank_away", "rank_change_home",
                    "total_points_home", "result", "rank_dif", "points_home_by_rank", "home_team_points"]]

    away_team = df[["date", "away_team", "away_score", "home_score", "rank_away", "rank_home", "rank_change_away",
                    "total_points_away", "result", "rank_dif", "points_away_by_rank", "away_team_points"]]
    home_team.columns = [h.replace("home_", "").replace("_home", "").replace("away_", "suf_").replace("_away", "_suf")
                         for h in home_team.columns]

    away_team.columns = [a.replace("away_", "").replace("_away", "").replace("home_", "suf_").replace("_home", "_suf")
                         for a in away_team.columns]
    team_stats = home_team.append(away_team)

    stats_val = []

    for index, row in team_stats.iterrows():
        team = row["team"]
        date = row["date"]
        past_games = team_stats.loc[
            (team_stats["team"] == team) & (team_stats["date"] < date)
            ].sort_values(by=['date'], ascending=False)
        last5 = past_games.head(5)

        goals = past_games["score"].mean()
        goals_l5 = last5["score"].mean()

        goals_suf = past_games["suf_score"].mean()
        goals_suf_l5 = last5["suf_score"].mean()

        rank = past_games["rank_suf"].mean()
        rank_l5 = last5["rank_suf"].mean()

        if len(last5) > 0:
            points = past_games["total_points"].values[0] - past_games["total_points"].values[
                -1]  # amount of points earned
            points_l5 = last5["total_points"].values[0] - last5["total_points"].values[-1]
        else:
            points = 0
            points_l5 = 0

        gp = past_games["team_points"].mean()
        gp_l5 = last5["team_points"].mean()

        gp_rank = past_games["points_by_rank"].mean()
        gp_rank_l5 = last5["points_by_rank"].mean()

        stats_val.append(
            [goals, goals_l5, goals_suf, goals_suf_l5, rank, rank_l5, points, points_l5, gp, gp_l5, gp_rank,
             gp_rank_l5])

    stats_cols = ["goals_mean", "goals_mean_l5", "goals_suf_mean", "goals_suf_mean_l5", "rank_mean", "rank_mean_l5",
                  "points_mean", "points_mean_l5", "game_points_mean", "game_points_mean_l5",
                  "game_points_rank_mean", "game_points_rank_mean_l5"]

    stats_df = pd.DataFrame(stats_val, columns=stats_cols)

    full_df = pd.concat([team_stats.reset_index(drop=True), stats_df], axis=1, ignore_index=False)

    home_team_stats = full_df.iloc[:int(full_df.shape[0] / 2), :]
    away_team_stats = full_df.iloc[int(full_df.shape[0] / 2):, :]

    home_team_stats = home_team_stats[home_team_stats.columns[-12:]]
    away_team_stats = away_team_stats[away_team_stats.columns[-12:]]

    home_team_stats.columns = ['home_' + str(col) for col in home_team_stats.columns]
    away_team_stats.columns = ['away_' + str(col) for col in away_team_stats.columns]

    # In order to unify the database, is needed to add home and away suffix for each column.
    # After that, the data is ready to be merged.
    match_stats = pd.concat([home_team_stats, away_team_stats.reset_index(drop=True)], axis=1, ignore_index=False)

    full_df = pd.concat([df, match_stats.reset_index(drop=True)], axis=1, ignore_index=False)

    # Drop friendly game
    full_df["is_friendly"] = full_df["tournament"].apply(lambda x: find_friendly(x))
    full_df = pd.get_dummies(full_df, columns=["is_friendly"])

    base_df = full_df[
        ["date", "home_team", "away_team", "rank_home", "rank_away", "home_score", "away_score", "result",
         "rank_dif", "rank_change_home", "rank_change_away", 'home_goals_mean',
         'home_goals_mean_l5', 'home_goals_suf_mean', 'home_goals_suf_mean_l5',
         'home_rank_mean', 'home_rank_mean_l5', 'home_points_mean',
         'home_points_mean_l5', 'away_goals_mean', 'away_goals_mean_l5',
         'away_goals_suf_mean', 'away_goals_suf_mean_l5', 'away_rank_mean',
         'away_rank_mean_l5', 'away_points_mean', 'away_points_mean_l5', 'home_game_points_mean',
         'home_game_points_mean_l5',
         'home_game_points_rank_mean', 'home_game_points_rank_mean_l5', 'away_game_points_mean',
         'away_game_points_mean_l5', 'away_game_points_rank_mean',
         'away_game_points_rank_mean_l5',
         'is_friendly_0', 'is_friendly_1']]

    df = base_df.dropna()

    df["target"] = df["result"].apply(lambda x: no_draw(x))

    model_db = create_db(df)

    return df, model_db


def find_friendly(x):
    """
    Return whether the match is friendly match or not.
    :param x:
    :return:
    """
    if x == "Friendly":
        return 1
    else:
        return 0


def create_db(df):
    """

    :param df:
    :return:
    """
    columns = ["home_team", "away_team", "target", "rank_dif", "home_goals_mean",
               "home_rank_mean", "away_goals_mean", "away_rank_mean", "home_rank_mean_l5", "away_rank_mean_l5",
               "home_goals_suf_mean", "away_goals_suf_mean", "home_goals_mean_l5", "away_goals_mean_l5",
               "home_goals_suf_mean_l5", "away_goals_suf_mean_l5", "home_game_points_rank_mean",
               "home_game_points_rank_mean_l5", "away_game_points_rank_mean", "away_game_points_rank_mean_l5",
               "is_friendly_0", "is_friendly_1"]

    base = df.loc[:, columns]
    base.loc[:, "goals_dif"] = base["home_goals_mean"] - base["away_goals_mean"]
    base.loc[:, "goals_dif_l5"] = base["home_goals_mean_l5"] - base["away_goals_mean_l5"]
    base.loc[:, "goals_suf_dif"] = base["home_goals_suf_mean"] - base["away_goals_suf_mean"]
    base.loc[:, "goals_suf_dif_l5"] = base["home_goals_suf_mean_l5"] - base["away_goals_suf_mean_l5"]
    base.loc[:, "goals_per_ranking_dif"] = (base["home_goals_mean"] / base["home_rank_mean"]) - (
            base["away_goals_mean"] / base["away_rank_mean"])
    base.loc[:, "dif_rank_agst"] = base["home_rank_mean"] - base["away_rank_mean"]
    base.loc[:, "dif_rank_agst_l5"] = base["home_rank_mean_l5"] - base["away_rank_mean_l5"]
    base.loc[:, "dif_points_rank"] = base["home_game_points_rank_mean"] - base["away_game_points_rank_mean"]
    base.loc[:, "dif_points_rank_l5"] = base["home_game_points_rank_mean_l5"] - base[
        "away_game_points_rank_mean_l5"]

    model_df = base[
        ["home_team", "away_team", "target", "rank_dif", "goals_dif", "goals_dif_l5",
         "goals_suf_dif", "goals_suf_dif_l5", "goals_per_ranking_dif", "dif_rank_agst", "dif_rank_agst_l5",
         "dif_points_rank", "dif_points_rank_l5", "is_friendly_0", "is_friendly_1"]]
    return model_df


def no_draw(x):
    """

    :param x:
    :return:
    """
    if x == 2:
        return 1
    else:
        return x
