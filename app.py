from flask import Flask

from cnn.views import cnn

def create_app():
    app = Flask(__name__)
    app.config.from_object('settings')

    app.register_blueprint(cnn)

    return app

