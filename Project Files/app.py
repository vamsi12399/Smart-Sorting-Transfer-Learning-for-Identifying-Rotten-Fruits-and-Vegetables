from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model/smart_sorting_model.h5')
class_names = ['Apple___Fresh','Apple___Rotten','Banana___Fresh','Banana___Rotten','Beetroot___Fresh','Beetroot___Rotten',
               'Bell Pepper___Fresh','Bell Pepper___Rotten','Cabbage___Fresh','Cabbage___Rotten','Carrot___Fresh',
               'Carrot___Rotten','Cauliflower___Fresh','Cauliflower___Rotten','Cucumber___Fresh','Cucumber___Rotten',
               'Garlic___Fresh','Garlic___Rotten','Ginger___Fresh','Ginger___Rotten','Grapes___Fresh','Grapes___Rotten',
               'Mango___Fresh','Mango___Rotten','Onion___Fresh','Onion___Rotten','Orange___Fresh','Orange___Rotten',
               'Potato___Fresh','Potato___Rotten','Tomato___Fresh','Tomato___Rotten']

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Image preprocessing
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    
    # Prepare image URL for display
    image_url = f"/uploads/{file.filename}"
    return render_template('index.html', prediction=predicted_class, image_url=image_url)

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
