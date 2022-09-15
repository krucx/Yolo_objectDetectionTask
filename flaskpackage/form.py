from flask_wtf import FlaskForm
from flask_wtf.file import FileField,FileRequired
from wtforms import SubmitField

class ImageUploadForm(FlaskForm):

    image = FileField('Upload Image',validators=[FileRequired()])
    submit = SubmitField('Get Objects!')

