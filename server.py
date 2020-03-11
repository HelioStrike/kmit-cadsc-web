from flask import Flask, flash, request, redirect, url_for, render_template, send_file
from markupsafe import escape
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import os
import io

app = Flask(__name__)

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/tasks/<task>', methods=['GET'])
def epithelium_segmentation(task):
    task = escape(task)
    return render_template('_'.join(task.split('-'))+'.html')

ALLOWED_IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'tif']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

@app.route('/process-image', methods=['POST'])
def process_image():
    if request.form['task'] == 'epithelium_segmentation':
        orig, mask = upload_file(['orig', 'mask'])
        lib = getattr(__import__('pipelines.'+request.form['task']), request.form['task'])
        display_image = lib.get_display_image(orig, mask)
        os.remove(orig)
        os.remove(mask)
    img = Image.fromarray(display_image.astype('uint8'))
    file_object = io.BytesIO()
    img.save(file_object, 'PNG')
    file_object.seek(0)
    return send_file(file_object, mimetype='image/PNG')

def upload_file(fnames):
    if request.method == 'POST':
        ret_names = []
        for fname in fnames:
            if fname not in request.files:
                flash(f'No {fname} part')
                return redirect(request.url)
            file = request.files[fname]
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                if not os.path.exists('tmp'):
                    os.mkdir('tmp')
                fname = os.path.join('tmp', filename)
                file.save(fname)
                ret_names.append(fname)
        return ret_names

if __name__=="__main__":
    app.run(host="0.0.0.0", port="5010")