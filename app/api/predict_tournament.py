from flask_restful import Resource, fields, marshal_with

from ml.model import base_df, ml_model
from ml.predictor import Predictor

response_model = {
    'result': fields.String,
}


class TournamentPredictApi(Resource):
    """
    A wrapper class API of head pose estimate
    """

    def __init__(self):
        self.predictor = Predictor(base_df, ml_model)

    @marshal_with(response_model)
    def get(self):
        result = self.predictor.predict_all_matches()

        return {
            'result': result,
        }
