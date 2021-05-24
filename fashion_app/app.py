import os
import sys
import logging
import uuid
from flask import Flask, flash, request, render_template, url_for, redirect, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from pathlib import Path
from mask_rcnn_matterport.model import Model
from mask_rcnn_matterport.visualize import display_instances, random_colors
import pandas as pd
from config import *
import skimage
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = SECRET_KEY

logging.basicConfig(level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
# load the model

def load_categories():
    with open(CATEGORIES_FILE) as json_file:
        data = json.load(json_file)
        return [element["name"] for element in data["categories"]]

model = Model()
categories = load_categories()

def save_to_csv(inputs):
    new_df = pd.DataFrame(inputs)
    if os.path.exists(PREDICTION_FILE):
        old_df = pd.read_pickle(PREDICTION_FILE)
        new_df = pd.concat([old_df, new_df]).reset_index(drop=True)
    new_df.to_pickle(PREDICTION_FILE)

def delete_csv(image_path):
    old_df = pd.read_pickle(PREDICTION_FILE)
    new_df = old_df[old_df["image_id"] != image_path.replace("\\", "/")]
    new_df.to_pickle(PREDICTION_FILE)

def render_image(str_image):
    class_names = ["bg"] + categories
    read_predictions=pd.read_pickle(PREDICTION_FILE)
    r = read_predictions[read_predictions["image_id"] == str_image.replace("\\", "/")].to_dict('records')[0]
    image = skimage.io.imread(str_image)
    return display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], colors=model.range_color)

@app.route('/')
@app.route('/<predictions>')
def index(predictions = "off"):
    sorted_images = sorted(Path(app.config['UPLOAD_FOLDER']).iterdir(), key=os.path.getmtime, reverse=True)
    images = {str(image):
        str(image) if predictions == "off" else render_image(str(image)) 
        for image in sorted_images if os.path.split(str(image))[1] not in IGNORE_UPLOAD_FILES
    }
    return render_template('index.html', images=images, box=predictions == "on")

def unique_filename(filename):
    _, ext = os.path.splitext(filename)
    return str(uuid.uuid4()) + ext

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    upload = request.form.get('upload', None)
    if request.method == 'POST':
        if upload:
            f = request.files['file']
            predictions = []
            im = Image.open(f)
            im.thumbnail(MAX_UPLOAD_IMAGE_SIZE, Image.ANTIALIAS)
            app.logger.info(f"Image size before loading {im.size}")
            filename = secure_filename(f.filename)
            filename = unique_filename(filename)
            complete_path_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            im.save(complete_path_file)
            predictions += model.detect(complete_path_file)
            save_to_csv(predictions)
    return redirect(url_for('index', predictions = request.form.get('predictions', "off")))


@app.route('/<picture>/<predictions>/', methods=['GET', 'POST'])
def delete_picture(picture, predictions):
    predictions = "on" if bool(predictions) is True else "off"
    delete_csv(picture)
    os.remove(picture)
    return redirect(url_for('index', predictions = predictions))

if __name__ == "__main__":
    app.run(debug=True)
