import os
from flask import Flask, request, redirect, flash, jsonify, render_template
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = 'supersecretkey'

# Load model and processor
processor = AutoImageProcessor.from_pretrained("imfarzanansari/skintelligent-acne")
model = AutoModelForImageClassification.from_pretrained("imfarzanansari/skintelligent-acne")

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(file_path)
                image = Image.open(file_path).convert('RGB')
                inputs = processor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                predictions = outputs.logits.argmax(-1).item()

                # Severity levels
                labels = [
                    'Clear Skin', 
                    'Occasional Spots', 
                    'Mild Acne', 
                    'Moderate Acne', 
                    'Severe Acne', 
                    'Very Severe Acne'
                ]
                num_classes = len(labels)

                if 0 <= predictions < num_classes:
                    acne_severity = labels[predictions]
                else:
                    acne_severity = 'Unknown Prediction'

                return jsonify({'acne_severity': acne_severity})
            except Exception as e:
                return jsonify({'error': f'Error processing image: {str(e)}'})
        else:
            flash('Invalid file format')
            return redirect(request.url)
    
    return render_template('index.html')

@app.route('/index/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
