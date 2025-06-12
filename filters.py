import cv2
import numpy as np
from utils import get_image_path, show_image

def vintage_effect(img):
    """Create vintage/retro effect"""
    # Add noise
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    vintage = cv2.add(img, noise)
    
    # Reduce saturation
    hsv = cv2.cvtColor(vintage, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 0.6  # Reduce saturation
    vintage = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Add vignette (dark edges)
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/2)
    kernel_y = cv2.getGaussianKernel(rows, rows/2)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    
    for i in range(3):
        vintage[:, :, i] = vintage[:, :, i] * mask
    
    return vintage.astype(np.uint8)

def oil_painting_effect(img, size=7, dynRatio=1):
    """Create oil painting effect"""
    return cv2.xphoto.oilPainting(img, size, dynRatio)

def pencil_sketch_effect(img):
    """Create pencil sketch effect"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invert the image
    inverted = 255 - gray
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(inverted, (111, 111), 0)
    
    # Invert the blurred image
    inverted_blur = 255 - blurred
    
    # Create pencil sketch
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    
    return sketch

def hdr_effect(img):
    """Create HDR (High Dynamic Range) effect"""
    return cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)

def summer_effect(img):
    """Create warm summer effect"""
    # Increase warmth (more yellow/orange)
    summer = img.copy().astype(np.float64)
    
    # Increase red and green channels slightly
    summer[:, :, 2] = np.clip(summer[:, :, 2] * 1.15, 0, 255)  # Red
    summer[:, :, 1] = np.clip(summer[:, :, 1] * 1.05, 0, 255)  # Green
    
    # Increase brightness
    summer = np.clip(summer + 10, 0, 255)
    
    return summer.astype(np.uint8)

def winter_effect(img):
    """Create cool winter effect"""
    # Add blue tint
    winter = img.copy().astype(np.float64)
    
    # Increase blue channel
    winter[:, :, 0] = np.clip(winter[:, :, 0] * 1.2, 0, 255)  # Blue
    
    # Slightly decrease red
    winter[:, :, 2] = np.clip(winter[:, :, 2] * 0.9, 0, 255)  # Red
    
    # Increase contrast
    winter = np.clip(winter * 1.1 - 10, 0, 255)
    
    return winter.astype(np.uint8)

def emboss_effect(img):
    """Create emboss effect"""
    kernel = np.array([[-2, -1, 0],
                      [-1,  1, 1],
                      [ 0,  1, 2]])
    
    embossed = cv2.filter2D(img, -1, kernel)
    
    # Convert to grayscale and add some color back
    gray = cv2.cvtColor(embossed, cv2.COLOR_BGR2GRAY)
    embossed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    return embossed

def neon_effect(img):
    """Create neon glow effect"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Create colored neon effect
    neon = np.zeros_like(img)
    
    # Make edges glow in different colors
    neon[:, :, 0] = edges  # Blue channel
    neon[:, :, 1] = edges * 0.5  # Green channel  
    neon[:, :, 2] = edges  # Red channel (creates purple/pink)
    
    # Add glow effect
    neon = cv2.GaussianBlur(neon, (15, 15), 0)
    
    # Combine with original
    result = cv2.addWeighted(img, 0.7, neon, 0.3, 0)
    
    return result

def interactive_filter_app():
    """Interactive filter application"""
    print("=== ADVANCED IMAGE FILTERS ===")
    print("Select an image to apply cool effects!")
    
    # Get image
    image_path = get_image_path("Select an image for filtering")
    if not image_path:
        print("No image selected!")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print("Could not load image!")
        return
    
    print(f"Loaded: {image_path}")
    
    # Show original
    show_image("Original Image", img)
    
    # Create effects dictionary
    effects = {
        '1': ('Vintage Effect', vintage_effect),
        '2': ('Pencil Sketch', pencil_sketch_effect),
        '3': ('HDR Effect', hdr_effect),
        '4': ('Summer Effect', summer_effect),
        '5': ('Winter Effect', winter_effect),
        '6': ('Emboss Effect', emboss_effect),
        '7': ('Neon Effect', neon_effect),
    }
    
    # Try to add oil painting (requires opencv-contrib-python)
    try:
        effects['8'] = ('Oil Painting', oil_painting_effect)
    except:
        print("⚠️ Oil painting effect not available (install opencv-contrib-python for this)")
    
    while True:
        print("\n" + "="*50)
        print("AVAILABLE EFFECTS:")
        for key, (name, _) in effects.items():
            print(f"'{key}' - {name}")
        print("'a' - Apply ALL effects (comparison view)")
        print("'s' - Save current effect")
        print("'r' - Reset to original")
        print("'q' - Quit")
        print("="*50)
        
        choice = input("Choose an effect: ").lower()
        
        if choice == 'q':
            break
        elif choice == 'r':
            show_image("Original Image", img)
            print("✅ Reset to original")
        elif choice == 's':
            # Save the currently displayed image
            save_path = image_path.replace('.', '_filtered.')
            # Note: This is simplified - in real app you'd track current effect
            print(f"💾 Save feature - implement based on last applied effect")
        elif choice == 'a':
            # Show all effects in a grid
            show_all_effects(img, effects)
        elif choice in effects:
            effect_name, effect_function = effects[choice]
            print(f"🎨 Applying {effect_name}...")
            
            try:
                if effect_name == "Pencil Sketch":
                    # Special handling for grayscale effect
                    result = effect_function(img)
                    result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                    show_image(f"{effect_name}", result_bgr)
                else:
                    result = effect_function(img)
                    show_image(f"{effect_name}", result)
                
                print(f"✅ {effect_name} applied!")
                
                # Ask if user wants to save
                save_choice = input("Save this effect? (y/n): ").lower()
                if save_choice == 'y':
                    save_path = image_path.replace('.', f'_{effect_name.lower().replace(" ", "_")}.')
                    if effect_name == "Pencil Sketch":
                        cv2.imwrite(save_path, result_bgr)
                    else:
                        cv2.imwrite(save_path, result)
                    print(f"💾 Saved: {save_path}")
                    
            except Exception as e:
                print(f"❌ Error applying {effect_name}: {e}")
        else:
            print("❌ Invalid choice!")
    
    cv2.destroyAllWindows()
    print("👋 Filter app closed!")

def show_all_effects(img, effects):
    """Show all effects in a comparison grid"""
    print("🎨 Generating all effects... this may take a moment...")
    
    # Create a large canvas for all effects
    effect_results = []
    
    # Add original
    original_small = cv2.resize(img, (300, 200))
    cv2.putText(original_small, 'Original', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    effect_results.append(original_small)
    
    # Apply each effect
    for key, (name, effect_function) in effects.items():
        try:
            if name == "Pencil Sketch":
                result = effect_function(img)
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            else:
                result = effect_function(img)
            
            # Resize for grid
            result_small = cv2.resize(result, (300, 200))
            
            # Add label
            cv2.putText(result_small, name, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            effect_results.append(result_small)
            
        except Exception as e:
            print(f"⚠️ Skipped {name}: {e}")
    
    # Create grid (3 columns)
    rows = []
    for i in range(0, len(effect_results), 3):
        row_images = effect_results[i:i+3]
        
        # Pad row if needed
        while len(row_images) < 3:
            blank = np.zeros((200, 300, 3), dtype=np.uint8)
            row_images.append(blank)
        
        row = np.hstack(row_images)
        rows.append(row)
    
    # Combine all rows
    if rows:
        grid = np.vstack(rows)
        show_image("All Effects Comparison", grid)
        
        # Ask if user wants to save the comparison
        save_choice = input("Save comparison grid? (y/n): ").lower()
        if save_choice == 'y':
            cv2.imwrite('effects_comparison.jpg', grid)
            print("💾 Saved: effects_comparison.jpg")

if __name__ == "__main__":
    interactive_filter_app()
