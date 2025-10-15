#!/usr/bin/env python3
"""
Simple STM32N6 Face Recognition Target Updater
A minimal version that captures one face and updates target_embedding.c
"""

import os
import sys
import time
import serial
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("❌ ONNXRuntime not available")
    sys.exit(1)

class SimpleFaceUpdater:
    def __init__(self):
        self.port = '/dev/ttyACM0'
        self.baudrate = 7372800
        self.serial = None
        self.buffer = b''
        self.embedding_session = None
        
        # Target file path
        self.target_file = Path(__file__).resolve().parents[1] / "embedded" / "Src" / "target_embedding.c"
        
    def connect(self):
        """Connect to STM32N6"""
        print(f"🔗 Connecting to {self.port}...")
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            print("✅ Connected!")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def load_model(self, model_path):
        """Load ONNX model"""
        print(f"🧠 Loading model...")
        try:
            self.embedding_session = ort.InferenceSession(model_path)
            print("✅ Model loaded!")
            return True
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def find_aln_frame(self, buffer):
        """Find ALN frame in buffer"""
        aln_pos = buffer.find(b'ALN')
        if aln_pos == -1:
            return None
        
        if aln_pos + 15 + 112*112 > len(buffer):
            return None
        
        try:
            header_end = aln_pos + 15
            data_end = header_end + 112*112
            image_data = buffer[header_end:data_end]
            image = np.frombuffer(image_data, dtype=np.uint8).reshape(112, 112)
            return image, data_end
        except:
            return None
    
    def analyze_quality(self, image):
        """Simple quality check"""
        brightness = float(np.mean(image))
        contrast = float(np.std(image))
        
        # Simple scoring
        score = 0
        if 60 <= brightness <= 200:
            score += 50
        if contrast >= 25:
            score += 50
            
        return score, brightness, contrast
    
    def extract_embedding(self, image):
        """Extract face embedding"""
        try:
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # Image is already 112x112, keep it that way
            
            # Normalize for model (float32, range 0-1)
            face_normalized = image_rgb.astype(np.float32) / 255.0
            face_input = np.transpose(face_normalized, (2, 0, 1))[None, ...]
            
            # Run inference
            input_name = self.embedding_session.get_inputs()[0].name
            output_name = self.embedding_session.get_outputs()[0].name
            onnx_output = self.embedding_session.run([output_name], {input_name: face_input})[0]
            
            # Normalize embedding
            embedding = onnx_output.astype(np.float32).flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
        except Exception as e:
            print(f"❌ Embedding extraction failed: {e}")
            return None
    
    def update_target_file(self, embedding):
        """Update target_embedding.c"""
        print(f"📝 Updating {self.target_file}...")
        
        try:
            # Read current file
            content = self.target_file.read_text()
            
            # Create new embedding values formatted nicely (8 values per line)
            embedding_values = [f"{x:.6f}f" for x in embedding]
            lines_of_values = []
            for i in range(0, len(embedding_values), 8):
                line_values = embedding_values[i:i+8]
                line_str = "    " + ", ".join(line_values)
                if i + 8 < len(embedding_values):
                    line_str += ","
                lines_of_values.append(line_str)
            
            # Create the new array declaration
            new_array = "float target_embedding[EMBEDDING_SIZE] = {\n" + \
                       "\n".join(lines_of_values) + "\n};"
            
            # Find the start and end of the current array
            lines = content.split('\n')
            start_idx = None
            end_idx = None
            
            for i, line in enumerate(lines):
                if 'float target_embedding[EMBEDDING_SIZE]' in line and '=' in line:
                    start_idx = i
                    # Find the closing brace
                    brace_count = line.count('{') - line.count('}')
                    j = i
                    while j < len(lines) and (brace_count > 0 or j == i):
                        if j > i:
                            brace_count += lines[j].count('{') - lines[j].count('}')
                        if brace_count <= 0 and lines[j].rstrip().endswith(';'):
                            end_idx = j
                            break
                        j += 1
                    break
            
            if start_idx is not None and end_idx is not None:
                # Replace the entire array declaration
                new_lines = lines[:start_idx] + [new_array] + lines[end_idx+1:]
                new_content = '\n'.join(new_lines)
                self.target_file.write_text(new_content)
                print("✅ Target embedding updated successfully!")
                return True
            else:
                print("❌ Could not find target_embedding array to replace")
                return False
                
        except Exception as e:
            print(f"❌ Failed to update target file: {e}")
            return False
    
    def capture_and_update(self, duration=10, min_quality=20):
        """Main capture loop"""
        print(f"🎯 Capturing for {duration} seconds (min quality: {min_quality})...")
        
        start_time = time.time()
        frames_processed = 0
        best_face = None
        best_score = 0
        
        while time.time() - start_time < duration:
            # Read data
            data = self.serial.read(4096)
            if data:
                self.buffer += data
            
            # Process frames
            while True:
                result = self.find_aln_frame(self.buffer)
                if not result:
                    break
                
                image, end_pos = result
                self.buffer = self.buffer[end_pos:]
                frames_processed += 1
                
                # Quality check
                score, brightness, contrast = self.analyze_quality(image)
                remaining = duration - (time.time() - start_time)
                
                print(f"📸 Frame {frames_processed} ({remaining:.1f}s left): "
                      f"Score={score}, Brightness={brightness:.0f}, Contrast={contrast:.0f}")
                
                # Keep best frame
                if score >= min_quality and score > best_score:
                    best_face = image.copy()
                    best_score = score
                    print(f"   ⭐ New best face! (score: {score})")
                
                # If we have a very good face, we can stop early
                if score >= 80:
                    print(f"   🎯 Excellent quality face found - stopping early!")
                    break
            
            # Keep buffer manageable
            if len(self.buffer) > 50000:
                self.buffer = self.buffer[-25000:]
            
            time.sleep(0.01)
        
        elapsed = time.time() - start_time
        print(f"\n📊 Capture complete: {elapsed:.1f}s, {frames_processed} frames")
        
        if best_face is not None:
            print(f"🎯 Processing best face (score: {best_score})...")
            
            # Extract embedding
            embedding = self.extract_embedding(best_face)
            if embedding is not None:
                print(f"🧠 Extracted embedding (size: {len(embedding)})")
                
                # Update target file
                if self.update_target_file(embedding):
                    print(f"🎉 SUCCESS! Target updated. Rebuild your STM32N6 firmware.")
                    return True
                else:
                    print(f"❌ Failed to update target file")
                    return False
            else:
                print(f"❌ Failed to extract embedding")
                return False
        else:
            print(f"❌ No suitable face found (min quality: {min_quality})")
            return False
    
    def disconnect(self):
        """Disconnect"""
        if self.serial:
            self.serial.close()
            print("🔌 Disconnected")

def main():
    # Simple command line handling
    duration = 10
    min_quality = 20
    model_path = "../input_models/mobilefacenet_int8_faces.onnx"
    
    if len(sys.argv) > 1:
        duration = int(sys.argv[1])
    if len(sys.argv) > 2:
        min_quality = int(sys.argv[2])
    
    updater = SimpleFaceUpdater()
    
    try:
        # Connect and setup
        if not updater.connect():
            return 1
        if not updater.load_model(model_path):
            return 1
        
        # Capture and update
        success = updater.capture_and_update(duration, min_quality)
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
        return 0
    finally:
        updater.disconnect()

if __name__ == "__main__":
    sys.exit(main())
