import pickle

from img2vec_pytorch import Img2Vec
from PIL import Image

#infer file
with open('./model.p', 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()

image_path = './dataset2/val/cloudy/cloudy54.jpg'

img = Image.open(image_path)

features = img2vec.get_vec(img)

predi = model.predict([features])

print(predi)