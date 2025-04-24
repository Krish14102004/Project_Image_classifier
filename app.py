from flask import Flask, render_template, request
from PIL import Image
import pickle
from img2vec_pytorch import Img2Vec
import os

app = Flask(__name__)
img2vec = Img2Vec()
model = pickle.load(open('./model.p', 'rb'))

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)
            img = Image.open(img_path).convert('RGB')
            features = img2vec.get_vec(img)
            prediction = model.predict([features])[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
