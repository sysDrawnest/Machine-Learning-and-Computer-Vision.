
import cv2
import time

class FaceDetector:
    def __init__(self):
        # Load the cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Alternative: Use LBP cascade (faster but slightly less accurate)
        # self.face_cascade = cv2.CascadeClassifier(
        #     cv2.data.haarcascades + 'lbpcascade_frontalface.xml'
        # )
        
        # Initialize variables for FPS calculation
        self.prev_time = 0
        self.curr_time = 0
        
    def calculate_fps(self):
        self.curr_time = time.time()
        fps = 1 / (self.curr_time - self.prev_time)
        self.prev_time = self.curr_time
        return fps
    
    def detect_faces_advanced(self):
        # Start video capture
        cap = cv2.VideoCapture(0)
        
        # Set camera properties (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Advanced Face Detection Started")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'd' - Toggle display mode")
        
        display_mode = 0  # 0: normal, 1: grayscale, 2: edge detection
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                maxSize=(300, 300)  # Maximum face size
            )
            
            # Create display frame based on mode
            if display_mode == 0:
                display_frame = frame.copy()
            elif display_mode == 1:
                display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            elif display_mode == 2:
                edges = cv2.Canny(gray, 100, 200)
                display_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Draw rectangles and face details
            for i, (x, y, w, h) in enumerate(faces):
                # Draw main rectangle
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw corner markers
                corner_length = 20
                # Top-left
                cv2.line(display_frame, (x, y), (x + corner_length, y), (255, 0, 0), 2)
                cv2.line(display_frame, (x, y), (x, y + corner_length), (255, 0, 0), 2)
                # Top-right
                cv2.line(display_frame, (x + w, y), (x + w - corner_length, y), (255, 0, 0), 2)
                cv2.line(display_frame, (x + w, y), (x + w, y + corner_length), (255, 0, 0), 2)
                # Bottom-left
                cv2.line(display_frame, (x, y + h), (x + corner_length, y + h), (255, 0, 0), 2)
                cv2.line(display_frame, (x, y + h), (x, y + h - corner_length), (255, 0, 0), 2)
                # Bottom-right
                cv2.line(display_frame, (x + w, y + h), (x + w - corner_length, y + h), (255, 0, 0), 2)
                cv2.line(display_frame, (x + w, y + h), (x + w, y + h - corner_length), (255, 0, 0), 2)
                
                # Add face number
                cv2.putText(display_frame, f'Face #{i+1}', (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add face dimensions
                cv2.putText(display_frame, f'{w}x{h}', (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Calculate and display FPS
            fps = self.calculate_fps()
            cv2.putText(display_frame, f'FPS: {fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display resolution
            cv2.putText(display_frame, f'Resolution: {width}x{height}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display mode
            mode_text = ['Normal', 'Grayscale', 'Edge Detection'][display_mode]
            cv2.putText(display_frame, f'Mode: {mode_text}', (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Advanced Face Detection', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                filename = f'face_detection_{timestamp}.jpg'
                cv2.imwrite(filename, display_frame)
                print(f"Screenshot saved as {filename}")
            elif key == ord('d'):
                display_mode = (display_mode + 1) % 3
                print(f"Switched to {mode_text} mode")
        
        cap.release()
        cv2.destroyAllWindows()

# Run the advanced version
if __name__ == "__main__":
    detector = FaceDetector()
    detector.detect_faces_advanced()
