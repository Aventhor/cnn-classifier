import os
from flask import Flask, request

from cnn.classifier import train


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./uploads"

@app.route("/train", methods=['GET'])
def train_model():
    return 'train'

@app.route("/", methods=['GET', 'POST'])
def classificate():
    file = request.files['file']
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img_class = train()
        return img_class

    return 'file not provided'
