from flask import Flask
from flask_restful import Api

from app.api.predict_match import MatchPredictApi
from app.api.predict_tournament import TournamentPredictApi
from configs.config import get_config


def create_app(config_object):
    """
    Create a Flask app.
    """
    app_ = Flask(__name__, instance_relative_config=True, static_folder='app/static')
    app_.app_context().push()
    api = Api(app_, prefix='/')
    api.add_resource(MatchPredictApi, '/match_predict', endpoint='match_predict')
    api.add_resource(TournamentPredictApi, '/tournament_predict', endpoint='tournament_predict')
    app_.config.from_object(config_object)
    return app_


if __name__ == '__main__':

    app = create_app(config_object=get_config("app_configs"))
    app.run(host='0.0.0.0', port=8686)
