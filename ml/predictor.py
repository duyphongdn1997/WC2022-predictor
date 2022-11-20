import os.path
from operator import itemgetter
from typing import Text, Tuple

import numpy as np
import pandas as pd

from configs.config import cfg
from configs.constants import DATA_ROOT
from ml.model import MLModel
from ml.utils import load_pickle


class Predictor:
    """
    A match predictor using ML
    """

    def __init__(self, base_df: pd.DataFrame, model: MLModel):
        self.model = model
        self.base_df = base_df

    def find_stats(self, team):
        """

        :param team: Name of the team, eg: Qatar, etc.
        :return:
        """

        last_game = self.base_df[(self.base_df["home_team"] == team) | (self.base_df["away_team"] == team)].tail(1)

        if last_game["home_team"].values[0] == team:
            team_rank = last_game["rank_home"].values[0]
            team_goals = last_game["home_goals_mean"].values[0]
            team_goals_l5 = last_game["home_goals_mean_l5"].values[0]
            team_goals_suf = last_game["home_goals_suf_mean"].values[0]
            team_goals_suf_l5 = last_game["home_goals_suf_mean_l5"].values[0]
            team_rank_suf = last_game["home_rank_mean"].values[0]
            team_rank_suf_l5 = last_game["home_rank_mean_l5"].values[0]
            team_gp_rank = last_game["home_game_points_rank_mean"].values[0]
            team_gp_rank_l5 = last_game["home_game_points_rank_mean_l5"].values[0]
        else:
            team_rank = last_game["rank_away"].values[0]
            team_goals = last_game["away_goals_mean"].values[0]
            team_goals_l5 = last_game["away_goals_mean_l5"].values[0]
            team_goals_suf = last_game["away_goals_suf_mean"].values[0]
            team_goals_suf_l5 = last_game["away_goals_suf_mean_l5"].values[0]
            team_rank_suf = last_game["away_rank_mean"].values[0]
            team_rank_suf_l5 = last_game["away_rank_mean_l5"].values[0]
            team_gp_rank = last_game["away_game_points_rank_mean"].values[0]
            team_gp_rank_l5 = last_game["away_game_points_rank_mean_l5"].values[0]

        return [team_rank, team_goals, team_goals_l5, team_goals_suf, team_goals_suf_l5, team_rank_suf,
                team_rank_suf_l5, team_gp_rank, team_gp_rank_l5]

    @staticmethod
    def find_features(team_1, team_2):
        """

        :param team_1:
        :param team_2:
        :return:
        """
        rank_dif = team_1[0] - team_2[0]
        goals_dif = team_1[1] - team_2[1]
        goals_dif_l5 = team_1[2] - team_2[2]
        goals_suf_dif = team_1[3] - team_2[3]
        goals_suf_dif_l5 = team_1[4] - team_2[4]
        goals_per_ranking_dif = (team_1[1] / team_1[5]) - (team_2[1] / team_2[5])
        dif_rank_agst = team_1[5] - team_2[5]
        dif_rank_agst_l5 = team_1[6] - team_2[6]
        dif_gp_rank = team_1[7] - team_2[7]
        dif_gp_rank_l5 = team_1[8] - team_2[8]

        return [rank_dif, goals_dif, goals_dif_l5, goals_suf_dif, goals_suf_dif_l5, goals_per_ranking_dif,
                dif_rank_agst, dif_rank_agst_l5, dif_gp_rank, dif_gp_rank_l5, 1, 0]

    def __predict(self, team_1: Text, team_2: Text):

        team_1_stat = self.find_stats(team_1)
        team_2_stat = self.find_stats(team_2)

        features_g1 = self.find_features(team_1_stat, team_2_stat)
        features_g2 = self.find_features(team_2_stat, team_1_stat)

        probs_g1 = self.model.predict_proba([features_g1])
        probs_g2 = self.model.predict_proba([features_g2])
        team_1_prob_g1 = probs_g1[0][0]
        team_1_prob_g2 = probs_g2[0][1]
        team_2_prob_g1 = probs_g1[0][1]
        team_2_prob_g2 = probs_g2[0][0]

        team_1_prob = (probs_g1[0][0] + probs_g2[0][1]) / 2
        team_2_prob = (probs_g2[0][0] + probs_g1[0][1]) / 2

        return team_1_prob_g1, team_1_prob_g2, team_1_prob, team_2_prob, team_2_prob_g1, team_2_prob_g2

    def predict(self, team_1: Text, team_2: Text) -> Tuple[bool, Text, float]:
        """

        :param team_1:
        :param team_2:
        :return:
        """
        draw = False
        team_1_prob_g1, team_1_prob_g2, team_1_prob, team_2_prob, team_2_prob_g1, team_2_prob_g2 = self.__predict(
            team_1, team_2)
        winner, winner_proba = "", 0.0
        if ((team_1_prob_g1 > team_2_prob_g1) & (team_2_prob_g2 > team_1_prob_g2)) | (
                (team_1_prob_g1 < team_2_prob_g1) & (team_2_prob_g2 < team_1_prob_g2)):
            draw = True

        elif team_1_prob > team_2_prob:
            winner = team_1
            winner_proba = team_1_prob

        elif team_2_prob > team_1_prob:
            winner = team_2
            winner_proba = team_2_prob
        return draw, winner, winner_proba

    def predict_all_matches(self) -> Text:
        """
        Predict all the matches in the tournament
        :return:
        """
        result = ""
        data = load_pickle(os.path.join(DATA_ROOT, cfg.data.table_matches))
        table = data['table']
        matches = data['matches']
        advanced_group, last_group = [], ""

        for teams in matches:
            draw = False
            team_1_prob_g1, team_1_prob_g2, team_1_prob, team_2_prob, team_2_prob_g1, team_2_prob_g2 = self.__predict(
                teams[1], teams[2])
            winner, winner_proba = "", 0.0
            if ((team_1_prob_g1 > team_2_prob_g1) & (team_2_prob_g2 > team_1_prob_g2)) | (
                    (team_1_prob_g1 < team_2_prob_g1) & (team_2_prob_g2 < team_1_prob_g2)):
                draw = True
                for i in table[teams[0]]:
                    if i[0] == teams[1] or i[0] == teams[2]:
                        i[1] += 1

            elif team_1_prob > team_2_prob:
                winner = teams[1]
                winner_proba = team_1_prob
                for i in table[teams[0]]:
                    if i[0] == teams[1]:
                        i[1] += 3

            elif team_2_prob > team_1_prob:
                winner = teams[2]
                winner_proba = team_2_prob
                for i in table[teams[0]]:
                    if i[0] == teams[2]:
                        i[1] += 3

            for i in table[teams[0]]:  # adding tiebreaker (probs per game)
                if i[0] == teams[1]:
                    i[2].append(team_1_prob)
                if i[0] == teams[2]:
                    i[2].append(team_2_prob)

            if last_group != teams[0]:
                if last_group != "":
                    result += "\n"
                    result += "Group %s advanced: \n" % last_group
                    for i in table[last_group]:  # adding tiebreaker
                        i[2] = np.mean(i[2])

                    final_points = table[last_group]
                    final_table = sorted(final_points, key=itemgetter(1, 2), reverse=True)
                    advanced_group.append([final_table[0][0], final_table[1][0]])
                    for i in final_table:
                        result += "%s -------- %d\n" % (i[0], i[1])
                result += "\n"
                result += "-" * 10 + " Starting Analysis for Group %s " % (teams[0]) + "-" * 10 + "\n"

            if draw is False:
                result += "Group %s - %s vs. %s: Winner %s with %.2f probability\n" % (
                    teams[0], teams[1], teams[2], winner, winner_proba)
            else:
                result += "Group %s - %s vs. %s: Draw\n" % (teams[0], teams[1], teams[2])
            last_group = teams[0]
        result += "\n"
        result += "Group %s advanced: \n" % last_group

        for i in table[last_group]:  # adding tiebreaker
            i[2] = np.mean(i[2])

        final_points = table[last_group]
        final_table = sorted(final_points, key=itemgetter(1, 2), reverse=True)
        advanced_group.append([final_table[0][0], final_table[1][0]])
        for i in final_table:
            result += "%s -------- %d\n" % (i[0], i[1])

        advanced = advanced_group
        playoffs = {"Round of 16": [], "Quarter-Final": [], "Semi-Final": [], "Final": []}

        for p in playoffs.keys():
            playoffs[p] = []

        actual_round = ""
        next_rounds = []

        for p in playoffs.keys():
            if p == "Round of 16":
                control = []
                for a in range(0, len(advanced * 2), 1):
                    if a < len(advanced):
                        if a % 2 == 0:
                            control.append((advanced * 2)[a][0])
                        else:
                            control.append((advanced * 2)[a][1])
                    else:
                        if a % 2 == 0:
                            control.append((advanced * 2)[a][1])
                        else:
                            control.append((advanced * 2)[a][0])
                playoffs[p] = [[control[c], control[c + 1]] for c in range(0, len(control) - 1, 1) if c % 2 == 0]

                for i in range(0, len(playoffs[p]), 1):
                    game = playoffs[p][i]

                    home = game[0]
                    away = game[1]

                    team_1_prob_g1, team_1_prob_g2, team_1_prob, team_2_prob, team_2_prob_g1, team_2_prob_g2 = \
                        self.__predict(home, away)
                    if actual_round != p:
                        result += "-" * 10 + "\n"
                        result += "Starting simulation of %s\n" % p
                        result += "-" * 10 + "\n"

                    if team_1_prob < team_2_prob:
                        result += "%s vs. %s: %s advances with prob %.2f\n" % (home, away, away, team_2_prob)
                        next_rounds.append(away)
                    else:
                        result += "%s vs. %s: %s advances with prob %.2f\n" % (home, away, home, team_1_prob)
                        next_rounds.append(home)

                    game.append([team_1_prob, team_2_prob])
                    playoffs[p][i] = game
                    actual_round = p

            else:
                playoffs[p] = [[next_rounds[c], next_rounds[c + 1]] for c in range(0, len(next_rounds) - 1, 1) if
                               c % 2 == 0]
                next_rounds = []
                for i in range(0, len(playoffs[p])):
                    game = playoffs[p][i]
                    home = game[0]
                    away = game[1]

                    team_1_prob_g1, team_1_prob_g2, team_1_prob, team_2_prob, team_2_prob_g1, team_2_prob_g2 = \
                        self.__predict(home, away)
                    if actual_round != p:
                        result += "-" * 10 + "\n"
                        result += "Starting simulation of %s\n" % p
                        result += "-" * 10 + "\n"

                    if team_1_prob < team_2_prob:
                        result += "%s vs. %s: %s advances with prob %.2f \n" % (home, away, away, team_2_prob)
                        next_rounds.append(away)
                    else:
                        result += "%s vs. %s: %s advances with prob %.2f \n" % (home, away, home, team_1_prob)
                        next_rounds.append(home)
                    game.append([team_1_prob, team_2_prob])
                    playoffs[p][i] = game
                    actual_round = p

        print(result)
        return result
