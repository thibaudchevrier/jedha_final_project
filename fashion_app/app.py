from flask import Flask, flash, request, render_template, url_for, redirect, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import os
import sys
import logging
import uuid
from pathlib import Path

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif'}
MAX_UPLOAD_IMAGE_SIZE = (400, 400)
IGNORE_UPLOAD_FILES = [".gitignore", "desktop.ini"]
SECRET_KEY = b'_5#y2L"F4Q8z\n\xec]/'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = SECRET_KEY

logging.basicConfig(level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

@app.route('/')
def index():
    sorted_images = sorted(Path(app.config['UPLOAD_FOLDER']).iterdir(), key=os.path.getmtime, reverse=True)
    images = [str(image) for image in sorted_images if os.path.split(image)[1] not in IGNORE_UPLOAD_FILES]
    app.logger.info(images)
    return render_template('index.html', images=images)

def unique_filename(filename):
    _, ext = os.path.splitext(filename)
    return ".".join([str(uuid.uuid4()), ext])

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        try:
            im = Image.open(f)
            im.thumbnail(MAX_UPLOAD_IMAGE_SIZE, Image.ANTIALIAS)
            app.logger.info(f"Image size before loading {im.size}")
            filename = secure_filename(f.filename)
            filename = unique_filename(filename)
            app.logger.info(f"Image size before loading {filename}")
            im.save(os.path.join(app.root_path.replace("\\","/"), app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('index'))
        except:
            flash("Incorrect file format !", 'error')
            return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True)
