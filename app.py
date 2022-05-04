import os
from flask import Flask, request

from cnn.classifier import train, classificate_image


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./uploads"

@app.route("/training", methods=['GET'])
def train_model():
    train()
    pass

@app.route("/predicting", methods=['POST'])
def classificate():
    file = request.files['file']
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img_class = classificate_image(filename)
        return img_class

    return 'file not provided'
