import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load pre-trained model (you can download a pre-trained emotion model)
# For this example, we'll create a simple placeholder
class SimpleEmotionDetector:
    def __init__(self):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
    def predict(self, face_img):
        # Placeholder - in reality, you'd use a real model
        return {
            'dominant_emotion': np.random.choice(self.emotions),
            'confidence': np.random.random()
        }

def main():
    # Initialize
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    detector = SimpleEmotionDetector()
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face
            face_roi = frame[y:y+h, x:x+w]
            
            # Predict emotion
            result = detector.predict(face_roi)
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{result['dominant_emotion']} ({result['confidence']:.2f})"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Emotion Detection', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()