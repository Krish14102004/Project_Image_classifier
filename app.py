from flask import Flask, render_template, request
from PIL import Image
import os
import pickle
from img2vec_pytorch import Img2Vec
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

with open('./model.p', 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None

    if request.method == 'POST':
        file = request.files['image']
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        img = Image.open(filepath)
        features = img2vec.get_vec(img)
        prediction = model.predict([features])[0]
        image_url = '/' + filepath.replace("\\", "/")

    return render_template('index.html', prediction=prediction, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
