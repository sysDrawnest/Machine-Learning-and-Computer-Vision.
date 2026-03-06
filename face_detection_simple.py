import cv2

def detect_faces():
    # Load the pre-trained face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Check if classifier loaded successfully
    if face_cascade.empty():
        print("Error loading face cascade classifier")
        return
    
    # Start video capture (0 for default webcam)
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Can't receive frame")
            break
        
        # Create a copy of the frame for display
        display_frame = frame.copy()
        
        # Convert frame to grayscale (required for face detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,    # How much the image size is reduced at each scale
            minNeighbors=5,      # How many neighbors each rectangle should have
            minSize=(30, 30),    # Minimum possible face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(display_frame, 'Face', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display number of faces detected
        cv2.putText(display_frame, f'Faces detected: {len(faces)}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow('Real-Time Face Detection', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):      # Quit
            break
        elif key == ord('s'):    # Save screenshot
            cv2.imwrite('face_detection_screenshot.jpg', display_frame)
            print("Screenshot saved!")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces()
