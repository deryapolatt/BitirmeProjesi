
import json
import os
from  flask_migrate import Migrate
from  flask_minify  import Minify
from   sys import exit
from flask import Flask, render_template, request, jsonify

from chatbot import predict_class, get_response
from apps.config import config_dict
from apps import create_app, db

# WARNING: Don't run with debug turned on in production!
DEBUG = (os.getenv('DEBUG', 'False') == 'True')

# The configuration
get_config_mode = 'Debug' if DEBUG else 'Production'

try:

    # Load the configuration using the default values
    app_config = config_dict[get_config_mode.capitalize()]

except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production] ')

app = create_app(app_config)
Migrate(app, db)

if not DEBUG:
    Minify(app=app, html=True, js=False, cssless=False)

@app.post("/predict")
def predict():
    intents = json.loads(open("intents.json", encoding="utf8").read())
    text = request.get_json().get("message")
    ints = predict_class(text)
    print(ints)
    response = get_response(ints, intents, text)
    message = {"answer": response}
    print(response)
    return jsonify(message)

if DEBUG:
    app.logger.info('DEBUG            = ' + str(DEBUG)             )
    app.logger.info('FLASK_ENV        = ' + os.getenv('FLASK_ENV') )
    app.logger.info('Page Compression = ' + 'FALSE' if DEBUG else 'TRUE' )
    app.logger.info('DBMS             = ' + app_config.SQLALCHEMY_DATABASE_URI)
    app.logger.info('ASSETS_ROOT      = ' + app_config.ASSETS_ROOT )

if __name__ == "__main__":
    app.run()
