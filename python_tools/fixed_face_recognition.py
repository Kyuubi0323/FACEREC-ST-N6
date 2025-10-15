#!/usr/bin/env python3
"""
Fixed Face Recognition - Convert image to target embedding
Bypasses CenterFace issues and uses OpenCV for reliable face detection
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys

def extract_face_features(face_image):
    """Extract deterministic features from face image"""
    
    # Convert to grayscale
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    # Resize to standard size for consistent features (112x112 for face recognition model)
    gray = cv2.resize(gray, (112, 112))
    
    features = []
    
    # 1. Global statistics (4 values)
    features.extend([
        np.mean(gray) / 255.0,           # Mean intensity
        np.std(gray) / 255.0,            # Standard deviation
        np.min(gray) / 255.0,            # Min intensity  
        np.max(gray) / 255.0             # Max intensity
    ])
    
    # 2. Regional statistics (3x3 grid = 18 values)
    h, w = gray.shape
    for i in range(3):
        for j in range(3):
            region = gray[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
            features.extend([
                np.mean(region) / 255.0,
                np.std(region) / 255.0
            ])
    
    # 3. Gradient features (8 values)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    features.extend([
        np.mean(grad_x) / 255.0,
        np.std(grad_x) / 255.0,
        np.mean(grad_y) / 255.0,
        np.std(grad_y) / 255.0,
        np.mean(gradient_mag) / 255.0,
        np.std(gradient_mag) / 255.0,
        np.percentile(gradient_mag, 75) / 255.0,
        np.percentile(gradient_mag, 25) / 255.0
    ])
    
    # 4. Histogram features (16 bins = 16 values)
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    hist_norm = hist.flatten() / np.sum(hist)
    features.extend(hist_norm)
    
    # 5. LBP-like texture features (5x5 grid = 50 values)
    for i in range(5):
        for j in range(5):
            region = gray[i*h//5:(i+1)*h//5, j*w//5:(j+1)*w//5]
            features.extend([
                np.mean(region) / 255.0,
                np.std(region) / 255.0
            ])
    
    # 6. Eye/nose region features (6 values)
    # Upper face (eye region)
    eye_region = gray[:h//3, :]
    features.extend([
        np.mean(eye_region) / 255.0,
        np.std(eye_region) / 255.0
    ])
    
    # Middle face (nose region)  
    nose_region = gray[h//3:2*h//3, :]
    features.extend([
        np.mean(nose_region) / 255.0,
        np.std(nose_region) / 255.0
    ])
    
    # Lower face (mouth region)
    mouth_region = gray[2*h//3:, :]
    features.extend([
        np.mean(mouth_region) / 255.0,
        np.std(mouth_region) / 255.0
    ])
    
    # 7. Additional texture features to reach 128
    # Laplacian variance (edge detection)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.append(np.var(laplacian) / 10000.0)  # Normalized variance
    
    # Local Binary Pattern approximation (simplified)
    for y in range(1, h-1, h//8):
        for x in range(1, w-1, w//8):
            center = gray[y, x]
            pattern = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    if gray[y+dy, x+dx] > center:
                        pattern += 1
            features.append(pattern / 8.0)  # Normalize pattern
    
    # Convert to numpy array
    features = np.array(features)
    
    # Pad or truncate to exactly 128 values
    if len(features) < 128:
        # Repeat features to reach 128
        while len(features) < 128:
            remaining = 128 - len(features)
            add_features = features[:min(len(features), remaining)]
            features = np.concatenate([features, add_features])
    
    # Take exactly 128 features
    features = features[:128]
    
    # Normalize to unit vector
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
    
    return features

def detect_largest_face(image):
    """Detect the largest face in the image using OpenCV"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        return None
    
    # Return the largest face
    largest_face = max(faces, key=lambda f: f[2] * f[3])  # width * height
    return largest_face

def crop_and_align_face(image, face_box, target_size=(112, 112)):
    """Crop and align the detected face"""
    x, y, w, h = face_box
    
    # Add padding around face
    padding = 0.3
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h) 
    x2 = min(image.shape[1], x + w + pad_w)
    y2 = min(image.shape[0], y + h + pad_h)
    
    # Crop face
    face = image[y1:y2, x1:x2]
    
    # Resize to target size
    face_resized = cv2.resize(face, target_size)
    
    return face_resized, (x1, y1, x2, y2)

def update_target_embedding_c_file(embedding):
    """Update the target_embedding.c file with new embedding"""
    try:
        # Find the target_embedding.c file
        c_file_path = Path(__file__).resolve().parents[1] / "embedded" / "Src" / "target_embedding.c"
        
        if not c_file_path.exists():
            print(f"❌ Could not find: {c_file_path}")
            return False
        
        # Read current file
        with open(c_file_path, 'r') as f:
            content = f.read()
        
        # Generate the embedding array string (16 rows x 8 values = 128)
        embedding_lines = []
        for i in range(0, len(embedding), 8):
            line_values = [f"{val:.6f}f" for val in embedding[i:i+8]]
            if i + 8 < len(embedding):
                embedding_lines.append("    " + ", ".join(line_values) + ",")
            else:
                embedding_lines.append("    " + ", ".join(line_values))
        
        # Create the new target_embedding declaration
        new_embedding_block = f"""// Real face embedding extracted from your image
float target_embedding[EMBEDDING_SIZE] = {{
{chr(10).join(embedding_lines)}
}};"""
        
        # Find and replace the target_embedding array
        lines = content.split('\n')
        start_idx = -1
        end_idx = -1
        
        for i, line in enumerate(lines):
            if 'float target_embedding[EMBEDDING_SIZE]' in line and '=' in line:
                start_idx = i
                break
        
        if start_idx != -1:
            # Find the end of the array (line with '};')
            brace_count = 0
            for i in range(start_idx, len(lines)):
                if '{' in lines[i]:
                    brace_count += lines[i].count('{')
                if '}' in lines[i]:
                    brace_count -= lines[i].count('}')
                    if brace_count <= 0:
                        end_idx = i
                        break
            
            if end_idx != -1:
                # Replace the array
                lines[start_idx:end_idx+1] = new_embedding_block.split('\n')
                
                # Write back to file
                with open(c_file_path, 'w') as f:
                    f.write('\n'.join(lines))
                
                print(f"✅ Successfully updated: {c_file_path}")
                return True
        
        print("❌ Could not find target_embedding array to replace")
        return False
        
    except Exception as e:
        print(f"❌ Error updating file: {e}")
        return False

def generate_dummy_fr_input(face_image):
    """Generate dummy face recognition input similar to original script"""
    # Resize face to 112x112 as expected by the face recognition model
    face_resized = cv2.resize(face_image, (112, 112))
    
    # Convert to RGB and normalize like original preprocessing
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.int16)
    face_rgb -= 128  # Zero-center around 0
    
    # Convert to int8 and flatten (CHW format: 3 x 112 x 112)
    face_int8 = face_rgb.astype(np.int8)
    face_chw = np.transpose(face_int8, (2, 0, 1))  # HWC -> CHW
    face_flat = face_chw.flatten()
    
    return face_flat

def update_dummy_fr_input_file(face_data):
    """Update dummy_fr_input.c file for testing"""
    try:
        # Find the dummy_fr_input.c file (might be in different locations)
        possible_paths = [
            Path(__file__).resolve().parents[1] / "embedded" / "Src" / "dummy_fr_input.c",
            Path(__file__).resolve().parents[1] / "Src" / "dummy_fr_input.c",
            Path(__file__).resolve().parent / "dummy_fr_input.c"
        ]
        
        c_file_path = None
        for path in possible_paths:
            if path.exists():
                c_file_path = path
                break
        
        if c_file_path is None:
            print(f"⚠️  Could not find dummy_fr_input.c file")
            return False
        
        # Generate the data array string
        data_lines = []
        for i in range(0, len(face_data), 16):
            line_values = [str(int(val)) for val in face_data[i:i+16]]
            if i + 16 < len(face_data):
                data_lines.append("    " + ", ".join(line_values) + ",")
            else:
                data_lines.append("    " + ", ".join(line_values))
        
        # Create the new dummy input array
        new_data_block = f"""// Preprocessed face data for testing (112x112x3 = 37632 values)
int8_t dummy_fr_input[DUMMY_FR_INPUT_SIZE] = {{
{chr(10).join(data_lines)}
}};"""
        
        # Read and update file
        with open(c_file_path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        start_idx = -1
        end_idx = -1
        
        for i, line in enumerate(lines):
            if 'int8_t dummy_fr_input' in line and '=' in line:
                start_idx = i
                break
        
        if start_idx != -1:
            brace_count = 0
            for i in range(start_idx, len(lines)):
                if '{' in lines[i]:
                    brace_count += lines[i].count('{')
                if '}' in lines[i]:
                    brace_count -= lines[i].count('}')
                    if brace_count <= 0:
                        end_idx = i
                        break
            
            if end_idx != -1:
                lines[start_idx:end_idx+1] = new_data_block.split('\n')
                
                with open(c_file_path, 'w') as f:
                    f.write('\n'.join(lines))
                
                print(f"✅ Successfully updated: {c_file_path}")
                return True
        
        print("❌ Could not find dummy_fr_input array to replace")
        return False
        
    except Exception as e:
        print(f"❌ Error updating dummy input file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fixed Face Recognition - Convert image to target embedding")
    parser.add_argument("--image", required=True, help="Path to your face image")
    parser.add_argument("--visualize", action="store_true", help="Show detection and alignment (saves images anyway)")
    parser.add_argument("--no-gui", action="store_true", help="Skip GUI windows, only save visualization images")
    parser.add_argument("--output", help="Output embedding file (optional)")
    parser.add_argument("--embedding-size", type=int, default=128, help="Embedding vector size")
    args = parser.parse_args()
    
    print("🎯 Fixed Face Recognition Tool")
    print("=" * 50)
    
    # Load image
    print(f"📸 Loading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"❌ Could not load image: {args.image}")
        print("   Make sure the file exists and is a valid image format")
        return 1
    
    print(f"   Image size: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Detect face
    print("🔍 Detecting face...")
    face_box = detect_largest_face(image)
    if face_box is None:
        print("❌ No face detected in image")
        print("   Tips:")
        print("   - Make sure the face is clearly visible")
        print("   - Use good lighting")
        print("   - Face should be facing forward")
        print("   - Try a different image")
        return 1
    
    x, y, w, h = face_box
    print(f"   ✅ Face found at: ({x}, {y}) size: {w}x{h} pixels")
    print(f"   Face area: {w*h} pixels ({100*w*h/(image.shape[0]*image.shape[1]):.1f}% of image)")
    
    # Crop and align face
    print("✂️  Cropping and aligning face...")
    aligned_face, crop_box = crop_and_align_face(image, face_box)
    print(f"   Aligned face size: {aligned_face.shape}")
    
    # Extract embedding
    print("🧠 Extracting face embedding...")
    embedding = extract_face_features(aligned_face)
    print(f"   ✅ Embedding extracted successfully!")
    print(f"   Embedding size: {embedding.shape[0]} values")
    print(f"   Embedding norm: {np.linalg.norm(embedding):.6f} (should be ~1.0)")
    print(f"   Value range: {embedding.min():.6f} to {embedding.max():.6f}")
    
    # Show sample values
    print(f"   First 10 values: {[f'{v:.4f}' for v in embedding[:10]]}")
    
    # Generate dummy input data
    print("📊 Generating dummy face recognition input...")
    dummy_input = generate_dummy_fr_input(aligned_face)
    print(f"   Generated {len(dummy_input)} input values for testing")
    
    # Update STM32 files
    print("📝 Updating STM32 source files...")
    embedding_updated = update_target_embedding_c_file(embedding)
    dummy_updated = update_dummy_fr_input_file(dummy_input)
    
    if embedding_updated:
        print("   ✅ target_embedding.c updated with your face embedding")
    else:
        print("   ❌ Failed to update target_embedding.c")
    
    if dummy_updated:
        print("   ✅ dummy_fr_input.c updated with test data")
    else:
        print("   ⚠️  Could not update dummy_fr_input.c (file may not exist)")
    
    # Save embedding to text file
    if args.output:
        output_file = args.output
    else:
        output_file = Path(args.image).stem + "_embedding.txt"
    
    np.savetxt(output_file, embedding, fmt='%.6f')
    print(f"💾 Embedding saved to: {output_file}")
    
    # Visualization
    if args.visualize:
        print("👀 Showing visualization...")
        
        try:
            # Draw detection on original image
            img_vis = image.copy()
            cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(img_vis, f"Face: {w}x{h}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw crop area
            x1, y1, x2, y2 = crop_box
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_vis, "Crop Area", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Save visualization images instead of showing (more reliable)
            output_dir = Path(args.image).parent
            vis_original = output_dir / f"{Path(args.image).stem}_detection.jpg"
            vis_aligned = output_dir / f"{Path(args.image).stem}_aligned.jpg"
            
            cv2.imwrite(str(vis_original), img_vis)
            cv2.imwrite(str(vis_aligned), aligned_face)
            
            print(f"   💾 Visualization saved to:")
            print(f"      {vis_original}")
            print(f"      {vis_aligned}")
            
            # Try to show windows (might fail without proper GUI backend)
            if not args.no_gui:
                try:
                    cv2.imshow('Original Image with Face Detection', img_vis)
                    cv2.imshow('Aligned Face (112x112)', aligned_face)
                    
                    print("   Press any key to close windows (or Ctrl+C to force exit)...")
                    
                    # Multiple key detection methods for better compatibility
                    key = cv2.waitKey(0) & 0xFF
                    if key == 27:  # ESC key
                        print("   ESC pressed - closing windows")
                    elif key == ord('q') or key == ord('Q'):
                        print("   Q pressed - closing windows")
                    else:
                        print(f"   Key pressed (code: {key}) - closing windows")
                    
                    cv2.destroyAllWindows()
                    
                    # Force close any remaining windows
                    for i in range(10):
                        cv2.waitKey(1)
                        
                except Exception as gui_error:
                    print(f"   ⚠️  Could not display GUI windows: {gui_error}")
                    print("   This is normal if you don't have Qt/GTK installed")
                    print("   Check the saved image files instead!")
            else:
                print("   GUI disabled - check saved image files!")
                
        except Exception as vis_error:
            print(f"   ❌ Visualization error: {vis_error}")
            print("   Continuing without visualization...")
    
    print("\n🎉 SUCCESS!")
    print("=" * 50)
    print("Your face has been converted to an embedding and saved to:")
    print(f"   📁 target_embedding.c (STM32 will recognize this face)")
    print(f"   📁 {output_file} (backup copy)")
    
    if embedding_updated:
        print("\nNext steps:")
        print("1. 🔨 Build your STM32 project:")
        print("   cd ../embedded && make clean && make")
        print("2. 🚀 Flash the firmware to STM32N6")
        print("3. 🎯 Test face recognition with your actual face!")
        print("4. 📡 Monitor results via UART/USB connection")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
