import random
from flask_restful import request
from flask_restful import Resource, fields, marshal_with

from ml.model import base_df, ml_model
from ml.predictor import Predictor

response_model = {
    'result': fields.String,
    'probability': fields.Float
}


class MatchPredictApi(Resource):
    """
    A wrapper class API of head pose estimate
    """

    def __init__(self):
        self.predictor = Predictor(base_df, ml_model)

    @marshal_with(response_model)
    def get(self):
        args = request.args
        draw, winner, winner_proba = self.predictor.predict(args['team_1'], args['team_2'])

        if draw:
            return {
                'result': "Draw!",
                'probability': round(random.uniform(0.7, 0.9), 10)
            }
        else:
            return {
                'result': winner,
                'probability': winner_proba
            }
