from flask import url_for,render_template,redirect,request,flash
from flaskpackage import app
from flaskpackage.form import ImageUploadForm
import os
import cv2

model = cv2.dnn.readNet(os.path.join(app.config['MODEL_FOLDER'],"yolov3.weights"), os.path.join(app.config['MODEL_FOLDER'],"yolov3.cfg"))

with open(os.path.join(app.config['MODEL_FOLDER'],"coco.names"), "r") as f:
    objects = [line.strip() for line in f.readlines()]

from flaskpackage.project import gen_bounding_boxes

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','jfif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/',methods=['GET','POST'])
@app.route('/home',methods=['GET','POST'])
def home():
    form = ImageUploadForm()
    if form.validate_on_submit():
        if allowed_file(form.image.data.filename):
            form.image.data.save(os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg'))
            if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], 'answer.jpg')):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], 'answer.jpg'))

            text = gen_bounding_boxes(model,objects)
            return render_template('result.html',text=text)
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif','danger')
    return render_template('home.html',form=form)


@app.route('/about')
def about():
    return render_template('about.html')