from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import base64

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('calibrated_svm_model.joblib')
scaler = joblib.load('scaler.joblib')

emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral', 'Contempt']

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None
    
    (x, y, w, h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    face_roi = cv2.resize(face_roi, (48, 48))
    face_roi = face_roi / 255.0
    face_roi = face_roi.flatten()
    
    return face_roi, (x, y, w, h)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_emotion():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Read the image file
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess the image
        result = preprocess_image(image)
        if result is None:
            return jsonify({'error': 'No face detected'})
        
        face_roi, (x, y, w, h) = result
        
        # Predict emotion
        face_roi_scaled = scaler.transform(face_roi.reshape(1, -1))
        emotion_probs = model.predict_proba(face_roi_scaled)[0]
        emotion_pred = np.argmax(emotion_probs)
        emotion = emotion_labels[emotion_pred]
        confidence = emotion_probs[emotion_pred] * 100
        
        # Draw rectangle and emotion on the image
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, f"{emotion}: {confidence:.2f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Convert image to base64 for displaying in HTML
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Get top 3 emotions
        top_3 = sorted(zip(emotion_labels, emotion_probs), key=lambda x: x[1], reverse=True)[:3]
        top_3 = [{'emotion': e, 'probability': float(p)} for e, p in top_3]
        
        return jsonify({
            'emotion': emotion,
            'confidence': float(confidence),
            'image': f'data:image/jpeg;base64,{img_base64}',
            'top_3': top_3
        })

if __name__ == '__main__':
    app.run(debug=True)