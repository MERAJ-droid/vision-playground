import cv2
import tkinter as tk
from tkinter import filedialog

def show_image(title, image, max_size=800):
    """Display image with automatic resizing"""
    height, width = image.shape[:2]
    if width > max_size or height > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    cv2.imshow(title, image)

def detect_faces():
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Get image from user (you already know this part!)
    root = tk.Tk()
    root.withdraw()
    
    image_path = filedialog.askopenfilename(
        title="Select a photo with faces",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if not image_path:
        print("No image selected")
        return
    
    # Load and process image (familiar territory!)
    img = cv2.imread(image_path)
    if img is None:
        print("Could not load image")
        return
    
    print(f"Processing: {image_path}")
    
    # Convert to grayscale (faces detect better in grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # This is the magic line - detect faces!
    faces = face_cascade.detectMultiScale(
        gray,           # Image to search
        scaleFactor=1.1,    # How much to reduce image size at each scale
        minNeighbors=5,     # How many neighbors each face needs
        minSize=(30, 30)    # Minimum face size
    )
    
    # Draw rectangles around detected faces
    img_with_faces = img.copy()
    for (x, y, w, h) in faces:
        # Draw rectangle
        cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add text label
        cv2.putText(img_with_faces, 'Face', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display results
    print(f"Found {len(faces)} face(s)!")
    show_image('Original Image', img)
    show_image('Faces Detected', img_with_faces)
    
    # Wait for key press
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Ask if user wants to save the result
    save = input("Save the result? (y/n): ")
    if save.lower() == 'y':
        output_path = image_path.replace('.', '_faces_detected.')
        cv2.imwrite(output_path, img_with_faces)
        print(f"Saved: {output_path}")

# Run the face detection
if __name__ == "__main__":
    print("=== Face Detection App ===")
    print("This will detect faces in your photos!")
    detect_faces()
