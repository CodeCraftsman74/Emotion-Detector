import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import joblib
import cv2

# Load the dataset
data = pd.read_csv(r"C:\Users\vijay babu\Downloads\ckextended.csv")  # Replace with your dataset path

def preprocess_data(data):
    data['pixels'] = data['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
    data['pixels'] = data['pixels'].apply(lambda x: x / 255.0)
    X = np.array(data['pixels'].tolist())
    y = data['emotion'].values
    return X, y

# Preprocess the dataset
X, y = preprocess_data(data)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
calibrated_svm = CalibratedClassifierCV(svm_model, cv=5, method='sigmoid')
calibrated_svm.fit(X_train_scaled, y_train)

# Test the model
y_pred = calibrated_svm.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and scaler
joblib.dump(calibrated_svm, 'calibrated_svm_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler have been saved.")

# Emotion labels
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral', 'Contempt']

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Print diagnostic information
print(f"OpenCV version: {cv2.__version__}")
print(f"Webcam width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"Webcam height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"Webcam FPS: {cap.get(cv2.CAP_PROP_FPS)}")

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    processed_faces = []
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi / 255.0
        face_roi = face_roi.flatten()
        processed_faces.append((face_roi, (x, y, w, h)))
    
    return processed_faces

print("Starting webcam capture. Close the 'Emotion Detection' window to quit.")

# Variables for simple tracking
prev_faces = []
max_faces = 5  # Maximum number of faces to track

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    processed_faces = preprocess_frame(frame)
    
    # Simple tracking: match new faces with previous faces
    if prev_faces:
        for i, (face_roi, face_coords) in enumerate(processed_faces):
            if i < len(prev_faces):
                prev_x, prev_y, prev_w, prev_h = prev_faces[i]
                x, y, w, h = face_coords
                # If the new face is close to the previous face, consider it the same face
                if abs(x - prev_x) < 50 and abs(y - prev_y) < 50:
                    face_coords = (
                        int(0.7 * prev_x + 0.3 * x),
                        int(0.7 * prev_y + 0.3 * y),
                        int(0.7 * prev_w + 0.3 * w),
                        int(0.7 * prev_h + 0.3 * h)
                    )
                    processed_faces[i] = (face_roi, face_coords)
    
    # Update prev_faces for the next iteration
    prev_faces = [face_coords for _, face_coords in processed_faces[:max_faces]]
    
    for face_roi, face_coords in processed_faces[:max_faces]:
        face_roi_scaled = scaler.transform(face_roi.reshape(1, -1))
        
        # Predict emotion probabilities
        emotion_probs = calibrated_svm.predict_proba(face_roi_scaled)[0]
        emotion_pred = np.argmax(emotion_probs)
        emotion = emotion_labels[emotion_pred]
        confidence = emotion_probs[emotion_pred] * 100
        
        # Draw rectangle around face and display emotion with confidence
        (x, y, w, h) = face_coords
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{emotion}: {confidence:.2f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display top 3 emotions for the first detected face
        if face_coords == processed_faces[0][1]:
            top_3 = sorted(zip(emotion_labels, emotion_probs), key=lambda x: x[1], reverse=True)[:3]
            for i, (emotion, prob) in enumerate(top_3):
                cv2.putText(frame, f"{emotion}: {prob*100:.2f}%", (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    if not processed_faces:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    cv2.imshow('Emotion Detection', frame)
    
    # Check if window has been closed
    if cv2.getWindowProperty('Emotion Detection', cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed. Quitting...")
        break
    
    # Wait for a short time and check for interrupts
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key
        print("ESC pressed. Quitting...")
        break

cap.release()
cv2.destroyAllWindows()

print("Webcam released and windows closed.")