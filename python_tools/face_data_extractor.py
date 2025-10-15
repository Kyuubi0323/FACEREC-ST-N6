#!/usr/bin/env python3
"""
STM32N6 Face Data Extractor
Shows what you can extract from the 112x112 face crops
"""

import serial
import numpy as np
import cv2
import time
import os
from datetime import datetime
from pathlib import Path

class FaceDataExtractor:
    def __init__(self, port='/dev/ttyACM0', baudrate=7372800):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.buffer = b''
        self.output_dir = "face_analysis"
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/faces", exist_ok=True)
        os.makedirs(f"{self.output_dir}/enhanced", exist_ok=True)
        
    def connect(self):
        """Connect to STM32N6"""
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            print(f"✅ Connected to {self.port}")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def extract_face_features(self, image):
        """Extract useful features from the face crop"""
        features = {}
        
        # Basic statistics
        features['brightness'] = float(np.mean(image))
        features['contrast'] = float(np.std(image))
        features['min_pixel'] = int(np.min(image))
        features['max_pixel'] = int(np.max(image))
        
        # Histogram analysis
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        features['histogram_peak'] = int(np.argmax(hist))
        features['dynamic_range'] = features['max_pixel'] - features['min_pixel']
        
        # Texture analysis (Laplacian variance for sharpness)
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        features['sharpness'] = float(laplacian.var())
        
        # Edge detection
        edges = cv2.Canny(image, 50, 150)
        features['edge_density'] = float(np.sum(edges > 0) / (image.shape[0] * image.shape[1]))
        
        # Face region analysis (assuming center region is face)
        center_h, center_w = image.shape[0] // 2, image.shape[1] // 2
        face_region = image[center_h-25:center_h+25, center_w-25:center_w+25]
        if face_region.size > 0:
            features['face_brightness'] = float(np.mean(face_region))
            features['face_contrast'] = float(np.std(face_region))
        
        # Symmetry analysis (compare left vs right half)
        mid = image.shape[1] // 2
        left_half = image[:, :mid]
        right_half = np.fliplr(image[:, mid:])
        if left_half.shape == right_half.shape:
            symmetry_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
            features['symmetry_score'] = float(255 - symmetry_diff)  # Higher = more symmetric
        
        return features
    
    def enhance_image(self, image):
        """Create enhanced versions of the face image"""
        enhanced = {}
        
        # Histogram equalization
        enhanced['equalized'] = cv2.equalizeHist(image)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced['clahe'] = clahe.apply(image)
        
        # Gaussian blur for noise reduction
        enhanced['denoised'] = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Sharpening kernel
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced['sharpened'] = cv2.filter2D(image, -1, kernel)
        enhanced['sharpened'] = np.clip(enhanced['sharpened'], 0, 255).astype(np.uint8)
        
        # Edge enhancement
        edges = cv2.Canny(image, 50, 150)
        enhanced['edges'] = edges
        
        # Combine original with edges
        edge_enhanced = cv2.addWeighted(image, 0.8, edges, 0.2, 0)
        enhanced['edge_enhanced'] = edge_enhanced
        
        return enhanced
    
    def save_analysis(self, image, features, enhanced, frame_id):
        """Save complete analysis to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_name = f"face_{timestamp}_f{frame_id:04d}"
        
        # Save original
        cv2.imwrite(f"{self.output_dir}/faces/{base_name}_original.png", image)
        
        # Save enhanced versions
        for name, enhanced_img in enhanced.items():
            cv2.imwrite(f"{self.output_dir}/enhanced/{base_name}_{name}.png", enhanced_img)
        
        # Save features as JSON
        import json
        features['timestamp'] = timestamp
        features['frame_id'] = frame_id
        features['filename'] = f"{base_name}_original.png"
        
        with open(f"{self.output_dir}/faces/{base_name}_features.json", 'w') as f:
            json.dump(features, f, indent=2)
        
        return base_name
    
    def analyze_face_quality(self, features):
        """Determine if this is a good quality face for recognition"""
        score = 0
        reasons = []
        
        # Brightness check (optimal 60-200)
        if 60 <= features['brightness'] <= 200:
            score += 25
        else:
            reasons.append(f"Poor brightness ({features['brightness']:.0f})")
        
        # Contrast check (minimum 25)
        if features['contrast'] >= 25:
            score += 25
        else:
            reasons.append(f"Low contrast ({features['contrast']:.0f})")
        
        # Sharpness check (minimum 100)
        if features['sharpness'] >= 100:
            score += 25
        else:
            reasons.append(f"Blurry ({features['sharpness']:.0f})")
        
        # Dynamic range check (minimum 100)
        if features['dynamic_range'] >= 100:
            score += 25
        else:
            reasons.append(f"Poor dynamic range ({features['dynamic_range']})")
        
        if score >= 75:
            quality = "EXCELLENT"
        elif score >= 50:
            quality = "GOOD"
        elif score >= 25:
            quality = "FAIR"
        else:
            quality = "POOR"
        
        return quality, score, reasons
    
    def find_aln_frame(self, buffer, start_pos=0):
        """Find and extract ALN frame"""
        aln_pos = buffer.find(b'ALN', start_pos)
        if aln_pos == -1:
            return None, len(buffer)
        
        if aln_pos + 20 > len(buffer):
            return None, aln_pos
        
        try:
            # ALN frame structure: ALN + header + 112x112 image data
            header_end = aln_pos + 15
            width, height = 112, 112  # Fixed size
            data_size = width * height
            data_end = header_end + data_size
            
            if data_end <= len(buffer):
                image_data = buffer[header_end:data_end]
                image_array = np.frombuffer(image_data, dtype=np.uint8)
                image_array = image_array.reshape((height, width))
                
                return {
                    'image': image_array,
                    'position': aln_pos,
                    'width': width,
                    'height': height
                }, data_end
            else:
                return None, aln_pos
        
        except Exception as e:
            print(f"⚠️ Error parsing frame: {e}")
            return None, aln_pos + 1
        
        return None, aln_pos + 1
    
    def run_analysis(self, max_frames=10, duration=30):
        """Run face data extraction and analysis"""
        if not self.serial:
            print("❌ Not connected")
            return
        
        print(f"🔍 Analyzing face data for {duration}s or {max_frames} frames...")
        print("📊 What we can extract from 112x112 face crops:")
        print("   • Brightness and contrast analysis")
        print("   • Sharpness and edge detection") 
        print("   • Face symmetry analysis")
        print("   • Image enhancement techniques")
        print("   • Quality scoring for face recognition")
        print()
        
        start_time = time.time()
        frames_analyzed = 0
        excellent_faces = 0
        good_faces = 0
        
        while (time.time() - start_time < duration and frames_analyzed < max_frames):
            # Read data
            data = self.serial.read(4096)
            if data:
                self.buffer += data
            
            # Process frames
            pos = 0
            while pos < len(self.buffer):
                frame_data, next_pos = self.find_aln_frame(self.buffer, pos)
                
                if frame_data:
                    image = frame_data['image']
                    frames_analyzed += 1
                    
                    print(f"📸 Frame {frames_analyzed}: {frame_data['width']}x{frame_data['height']}")
                    
                    # Extract features
                    features = self.extract_face_features(image)
                    
                    # Analyze quality
                    quality, score, reasons = self.analyze_face_quality(features)
                    
                    print(f"   Quality: {quality} (Score: {score}/100)")
                    print(f"   Brightness: {features['brightness']:.0f}, Contrast: {features['contrast']:.0f}")
                    print(f"   Sharpness: {features['sharpness']:.0f}, Symmetry: {features.get('symmetry_score', 0):.0f}")
                    
                    if reasons:
                        print(f"   Issues: {', '.join(reasons)}")
                    
                    # Create enhanced versions
                    enhanced = self.enhance_image(image)
                    
                    # Save analysis if quality is good
                    if score >= 50:  # Good or excellent
                        saved_name = self.save_analysis(image, features, enhanced, frames_analyzed)
                        print(f"   💾 Saved analysis: {saved_name}")
                        
                        if quality == "EXCELLENT":
                            excellent_faces += 1
                        elif quality == "GOOD":
                            good_faces += 1
                    
                    print()
                    pos = next_pos
                else:
                    pos = next_pos
                    if pos >= len(self.buffer):
                        break
            
            # Manage buffer size
            if len(self.buffer) > 65536:
                self.buffer = self.buffer[-32768:]
                pos = 0
            
            time.sleep(0.01)
        
        elapsed = time.time() - start_time
        print(f"📊 Analysis Complete:")
        print(f"   ⏱️  Duration: {elapsed:.1f} seconds")
        print(f"   📸 Frames analyzed: {frames_analyzed}")
        print(f"   ⭐ Excellent quality: {excellent_faces}")
        print(f"   ✅ Good quality: {good_faces}")
        print(f"   📁 Data saved to: {os.path.abspath(self.output_dir)}")
        
        if frames_analyzed > 0:
            print(f"\n🎯 Extraction Summary:")
            print(f"   • Face crops: 112x112 grayscale images")
            print(f"   • Features: Brightness, contrast, sharpness, symmetry")
            print(f"   • Enhancements: CLAHE, denoising, sharpening, edge detection")
            print(f"   • Quality scoring: For face recognition suitability")
            print(f"   • Metadata: JSON files with all measurements")
    
    def disconnect(self):
        if self.serial:
            self.serial.close()
            print("🔌 Disconnected")

def main():
    extractor = FaceDataExtractor()
    
    if extractor.connect():
        try:
            extractor.run_analysis(max_frames=5, duration=20)
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        finally:
            extractor.disconnect()

if __name__ == "__main__":
    main()
