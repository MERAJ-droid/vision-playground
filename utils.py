import cv2

def show_image(title, image, max_size=800):
    """Display image with automatic resizing - reusable function"""
    height, width = image.shape[:2]
    if width > max_size or height > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    cv2.imshow(title, image)

def get_image_path(title="Select an image"):
    """Get image path using file dialog - reusable function"""
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()
    
    image_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    
    return image_path
