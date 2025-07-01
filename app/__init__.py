from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__, static_folder='static')
    app.secret_key = 'your_super_secret_key'
    CORS(app)

    from .routes import main as main_routes
    app.register_blueprint(main_routes)

    return app
