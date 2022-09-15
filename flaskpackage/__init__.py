from flask import Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'flaskpackage/static/uploads/'
app.config['SECRET_KEY'] = '760b7067b089e56c762af9d24ce6b059'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] ='flaskpackage/static/'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
from flaskpackage import routes 