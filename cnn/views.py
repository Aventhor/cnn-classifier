import os

from flask import Blueprint, request, current_app

from cnn.classifier import classificate_image, train


cnn = Blueprint('cnn', __name__, url_prefix='/cnn')

@cnn.route("/training")
def train_model():
    train()
    pass

@cnn.route("/predicting", methods=['POST'])
def classificate():
    file = request.files['file']
    if file:
        filename = file.filename
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename) 
        file.save(filepath)
        img_class = classificate_image(filename)
        return { 'predict': img_class, 'path': filepath }

    return 'file not provided'