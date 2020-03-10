from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/epithelium-segmentation')
def epithelium_segmentation():
    return render_template('epithelium_segmentation.html')

ALLOWED_IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'tif']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

@app.route('/process-image', methods=['POST'])
def process_image():
    fname = upload_file()
    lib = getattr(__import__('pipelines.'+request.form['task']), request.form['task'])
    display_image = lib.get_display_image(fname)
    os.remove(fname)
    return display_image

def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not os.path.exists('tmp'):
                os.mkdir('tmp')
            fname = os.path.join('tmp', filename)
            file.save(fname)
            return fname

if __name__=="__main__":
    app.run(host="0.0.0.0", port="5010", debug=True)