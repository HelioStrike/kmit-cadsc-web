from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/epithelium-segmentation')
def epithelium_segmentation():
    return render_template('epithelium_segmentation.html')

if __name__=="__main__":
    app.run(host="0.0.0.0", port="5010", debug=True)