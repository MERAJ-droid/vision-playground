import cv2
import numpy as np

def show_instructions():
    """Display control instructions"""
    print("=== WEBCAM EFFECTS ===")
    print("Controls:")
    print("'n' - Normal view")
    print("'g' - Grayscale")
    print("'b' - Blur effect")
    print("'e' - Edge detection")
    print("'s' - Sepia effect")
    print("'c' - Cartoon effect")
    print("'f' - Face detection (live!)")
    print("'SPACE' - Take screenshot")
    print("'q' - Quit")
    print("\nStarting webcam...")

def apply_sepia(img):
    """Apply sepia effect"""
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                            [0.349, 0.686, 0.168],
                            [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, sepia_filter)
    return np.clip(sepia, 0, 255).astype(np.uint8)

def apply_cartoon(img):
    """Apply cartoon effect"""
    # Reduce colors
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    cartoon = centers[labels.flatten()].reshape(img.shape)
    
    # Add edge lines
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return cv2.bitwise_and(cartoon, edges)

def webcam_effects():
    """Main webcam effects function"""
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        print("Make sure your webcam is connected and not being used by another app")
        return
    
    # Load face detection model
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except:
        print("⚠️ Face detection model not loaded")
        face_cascade = None
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    current_effect = 'normal'
    screenshot_count = 0
    
    show_instructions()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        # Flip frame horizontally (mirror effect)
        frame = cv2.flip(frame, 1)
        
        # Apply selected effect
        if current_effect == 'normal':
            processed_frame = frame.copy()
            
        elif current_effect == 'gray':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
        elif current_effect == 'blur':
            processed_frame = cv2.GaussianBlur(frame, (15, 15), 0)
            
        elif current_effect == 'edge':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            processed_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
        elif current_effect == 'sepia':
            processed_frame = apply_sepia(frame)
            
        elif current_effect == 'cartoon':
            processed_frame = apply_cartoon(frame)
            
        elif current_effect == 'face' and face_cascade is not None:
            processed_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(processed_frame, 'Face Detected!', (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            processed_frame = frame.copy()
        
        # Add current effect text
        cv2.putText(processed_frame, f'Effect: {current_effect.upper()}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show the frame
        cv2.imshow('Webcam Effects - Press Q to quit', processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_effect = 'normal'
            print("✅ Normal view")
        elif key == ord('g'):
            current_effect = 'gray'
            print("✅ Grayscale effect")
        elif key == ord('b'):
            current_effect = 'blur'
            print("✅ Blur effect")
        elif key == ord('e'):
            current_effect = 'edge'
            print("✅ Edge detection")
        elif key == ord('s'):
            current_effect = 'sepia'
            print("✅ Sepia effect")
        elif key == ord('c'):
            current_effect = 'cartoon'
            print("✅ Cartoon effect")
        elif key == ord('f'):
            current_effect = 'face'
            print("✅ Live face detection")
        elif key == ord(' '):  # Spacebar
            screenshot_count += 1
            filename = f'screenshot_{screenshot_count}.jpg'
            cv2.imwrite(filename, processed_frame)
            print(f"📸 Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Webcam closed!")

if __name__ == "__main__":
    webcam_effects()
