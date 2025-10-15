#!/usr/bin/env python3
"""
STM32N6 Face Recognition Target Updater
Live capture faces from STM32N6 and update target_embedding.c for face recognition

This script:
1. Connects to STM32N6 via ST-Link VCP
2. Captures live face crops (112x112)
3. Processes them for face recognition 
4. Updates target_embedding.c with new face embeddings
5. Enables your STM32N6 system to recognize specific faces
"""

import serial
import numpy as np
import cv2
import time
import os
import json
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import argparse
import sys

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ ONNXRuntime not available - face embedding extraction disabled")

class STM32FaceRecognitionUpdater:
    def __init__(self, port='/dev/ttyACM0', baudrate=7372800):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.buffer = b''
        
        # Timeout settings
        self.connection_timeout = 10.0  # 10 seconds to connect
        self.frame_timeout = 5.0        # 5 seconds between frames
        self.model_timeout = 30.0       # 30 seconds to load model
        self.capture_active = False     # Flag for active capture
        
        # Face recognition setup
        self.face_recognition_model = None
        self.embedding_session = None
        
        # Data storage
        self.output_dir = "face_recognition_data"
        self.captured_faces = []
        self.face_embeddings = []
        
        # STM32N6 integration paths
        self.target_embedding_path = Path(__file__).resolve().parents[1] / "embedded" / "Src" / "target_embedding.c"
        self.dummy_input_path = Path(__file__).resolve().parents[1] / "embedded" / "Src" / "dummy_fr_input.c"
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/faces", exist_ok=True)
        os.makedirs(f"{self.output_dir}/processed", exist_ok=True)
        
    def connect(self):
        """Connect to STM32N6 with timeout"""
        print(f"🔗 Connecting to STM32N6 on {self.port}...")
        
        start_time = time.time()
        while time.time() - start_time < self.connection_timeout:
            try:
                self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
                print(f"✅ Connected to STM32N6 on {self.port}")
                
                # Test connection by trying to read some data
                print("📡 Testing connection...")
                test_start = time.time()
                data_received = False
                
                while time.time() - test_start < 3.0:  # 3 second test
                    data = self.serial.read(1024)
                    if data:
                        data_received = True
                        print(f"✅ Connection verified - receiving data ({len(data)} bytes)")
                        break
                    time.sleep(0.1)
                
                if not data_received:
                    print("⚠️ Warning: No data received in 3 seconds - check STM32N6 status")
                    # Don't fail here, maybe the board just started
                
                return True
                
            except serial.SerialException as e:
                if "Permission denied" in str(e):
                    print(f"❌ Permission denied - try: sudo chmod 666 {self.port}")
                    return False
                elif "No such file or directory" in str(e):
                    print(f"❌ Port not found: {self.port}")
                    return False
                else:
                    print(f"⚠️ Connection attempt failed: {e}")
                    time.sleep(1)
            except Exception as e:
                print(f"⚠️ Unexpected error: {e}")
                time.sleep(1)
        
        print(f"❌ Failed to connect after {self.connection_timeout} seconds")
        return False
    
    def load_face_recognition_model(self, model_path="input_models/mobilefacenet_int8_faces.onnx"):
        """Load the face recognition ONNX model with timeout"""
        if not ONNX_AVAILABLE:
            print("❌ ONNXRuntime not available - cannot load face recognition model")
            return False
        
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"❌ Face recognition model not found: {model_path}")
            print("   Please ensure the model file exists or run model conversion first")
            return False
        
        print(f"🧠 Loading face recognition model...")
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Model loading timeout")
        
        try:
            # Set up timeout for model loading
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.model_timeout))
            
            self.embedding_session = ort.InferenceSession(str(model_path))
            
            # Cancel timeout
            signal.alarm(0)
            
            print(f"✅ Loaded face recognition model: {model_path}")
            return True
            
        except TimeoutError:
            print(f"❌ Model loading timeout ({self.model_timeout}s) - model may be corrupted")
            return False
        except Exception as e:
            # Cancel timeout in case of other errors
            signal.alarm(0)
            print(f"❌ Failed to load face recognition model: {e}")
            return False
        finally:
            # Ensure timeout is always cancelled
            signal.alarm(0)
    
    def find_aln_frame(self, buffer, start_pos=0):
        """Find and extract ALN frame from STM32N6 data"""
        aln_pos = buffer.find(b'ALN', start_pos)
        if aln_pos == -1:
            return None, len(buffer)
        
        if aln_pos + 20 > len(buffer):
            return None, aln_pos
        
        try:
            # ALN frame structure: ALN + header + 112x112 image data
            header_end = aln_pos + 15
            width, height = 112, 112  # Fixed size from STM32N6
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
                    'height': height,
                    'timestamp': time.time()
                }, data_end
            else:
                return None, aln_pos
        
        except Exception as e:
            print(f"⚠️ Error parsing ALN frame: {e}")
            return None, aln_pos + 1
        
        return None, aln_pos + 1
    
    def analyze_face_quality(self, image):
        """Analyze face quality for recognition suitability"""
        features = {}
        
        # Basic statistics
        features['brightness'] = float(np.mean(image))
        features['contrast'] = float(np.std(image))
        features['min_pixel'] = int(np.min(image))
        features['max_pixel'] = int(np.max(image))
        features['dynamic_range'] = features['max_pixel'] - features['min_pixel']
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        features['sharpness'] = float(laplacian.var())
        
        # Face quality scoring
        score = 0
        reasons = []
        
        # Brightness check (optimal 60-200)
        if 60 <= features['brightness'] <= 200:
            score += 25
        else:
            reasons.append(f"brightness({features['brightness']:.0f})")
        
        # Contrast check (minimum 25)
        if features['contrast'] >= 25:
            score += 25
        else:
            reasons.append(f"contrast({features['contrast']:.0f})")
        
        # Sharpness check (minimum 100)
        if features['sharpness'] >= 100:
            score += 25
        else:
            reasons.append(f"sharpness({features['sharpness']:.0f})")
        
        # Dynamic range check (minimum 80)
        if features['dynamic_range'] >= 80:
            score += 25
        else:
            reasons.append(f"range({features['dynamic_range']})")
        
        features['quality_score'] = score
        features['quality_reasons'] = reasons
        
        if score >= 75:
            features['quality'] = "EXCELLENT"
            features['suitable_for_recognition'] = True
        elif score >= 50:
            features['quality'] = "GOOD"
            features['suitable_for_recognition'] = True
        else:
            features['quality'] = "POOR"
            features['suitable_for_recognition'] = False
        
        return features
    
    def preprocess_for_recognition(self, image):
        """Preprocess 112x112 grayscale image for face recognition model"""
        try:
            # Convert grayscale to RGB (model expects RGB)
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image
            
            # Resize to model input size (96x112)
            # STM32N6 gives us 112x112, model expects 96x112
            if image_rgb.shape != (112, 96, 3):
                image_resized = cv2.resize(image_rgb, (96, 112))
            else:
                image_resized = image_rgb
            
            # Convert BGR to RGB if needed
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize for model: zero-center around 0, int8 format
            face_normalized = image_rgb.astype(np.int16) - 128
            face_int8 = face_normalized.astype(np.int8)
            
            # CHW format and add batch dimension
            face_input = np.transpose(face_int8, (2, 0, 1))[None, ...]
            
            return face_input
            
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            return None
    
    def extract_face_embedding(self, image):
        """Extract face embedding using the loaded model"""
        if not self.embedding_session:
            print("❌ Face recognition model not loaded")
            return None
        
        try:
            # Preprocess image
            processed_image = self.preprocess_for_recognition(image)
            if processed_image is None:
                return None
            
            # Run inference
            input_name = self.embedding_session.get_inputs()[0].name
            output_name = self.embedding_session.get_outputs()[0].name
            
            onnx_output = self.embedding_session.run([output_name], {input_name: processed_image})[0]
            
            # Normalize embedding
            embedding = onnx_output.astype(np.float32).flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            print(f"❌ Embedding extraction error: {e}")
            return None
    
    def save_face_data(self, image, features, embedding, face_id):
        """Save face data to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_name = f"face_{timestamp}_id{face_id:03d}_q{features['quality_score']:02d}"
        
        # Save original image
        face_path = Path(self.output_dir) / "faces" / f"{base_name}.png"
        cv2.imwrite(str(face_path), image)
        
        # Save processed image (if we have embedding)
        if embedding is not None:
            # Create visualization
            processed_image = self.preprocess_for_recognition(image)
            if processed_image is not None:
                # Convert back to displayable format
                display_img = processed_image[0].transpose(1, 2, 0) + 128
                display_img = np.clip(display_img, 0, 255).astype(np.uint8)
                processed_path = Path(self.output_dir) / "processed" / f"{base_name}_processed.png"
                cv2.imwrite(str(processed_path), display_img)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'face_id': face_id,
            'features': features,
            'embedding_available': embedding is not None,
            'embedding_size': len(embedding) if embedding is not None else 0,
            'original_image': str(face_path),
            'source': 'STM32N6_live_capture'
        }
        
        if embedding is not None:
            metadata['embedding'] = embedding.tolist()
        
        metadata_path = Path(self.output_dir) / "faces" / f"{base_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return base_name, face_path, metadata_path
    
    def update_target_embedding(self, embedding, face_info):
        """Update target_embedding.c with new embedding"""
        if not self.target_embedding_path.exists():
            print(f"❌ Target embedding file not found: {self.target_embedding_path}")
            return False
        
        try:
            # Read current file
            lines = self.target_embedding_path.read_text().splitlines(keepends=True)
            
            # Create new embedding line
            embedding_values = ", ".join(f"{x:.6f}f" for x in embedding)
            new_line = f"float target_embedding[EMBEDDING_SIZE] = {{{embedding_values}}};\n"
            
            # Find and replace the embedding line
            updated = False
            for i, line in enumerate(lines):
                if line.strip().startswith("float target_embedding"):
                    lines[i] = new_line
                    updated = True
                    print(f"✅ Updated embedding at line {i+1}")
                    break
            
            if updated:
                # Write back to file
                self.target_embedding_path.write_text("".join(lines))
                
                # Add comment with face info
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                comment_lines = [
                    f"// Updated on {timestamp}\n",
                    f"// Face ID: {face_info.get('face_id', 'unknown')}\n",
                    f"// Quality: {face_info.get('quality', 'unknown')}\n",
                    f"// Source: STM32N6 live capture\n",
                    "\n"
                ]
                
                # Insert comments after the embedding
                for i, line in enumerate(lines):
                    if line.strip().startswith("float target_embedding"):
                        lines[i:i+1] = [line] + comment_lines
                        break
                
                self.target_embedding_path.write_text("".join(lines))
                print(f"✅ Successfully updated {self.target_embedding_path}")
                return True
            else:
                print("❌ Could not find target_embedding declaration to update")
                return False
                
        except Exception as e:
            print(f"❌ Error updating target embedding: {e}")
            return False
    
    def capture_and_update_target(self, capture_duration=30, min_quality_score=75):
        """Capture faces from STM32N6 and update target embedding with timeout handling"""
        if not self.serial:
            print("❌ Not connected to STM32N6")
            return False
        
        print(f"🎯 Capturing target face for recognition...")
        print(f"   Duration: {capture_duration} seconds")
        print(f"   Minimum quality: {min_quality_score}/100")
        print(f"   Frame timeout: {self.frame_timeout} seconds")
        print(f"   Look at the camera and stay still for best results!")
        print()
        
        self.capture_active = True
        start_time = time.time()
        last_frame_time = time.time()
        frames_processed = 0
        suitable_faces = []
        no_data_count = 0
        max_no_data = 50  # Maximum consecutive reads with no data
        
        try:
            while (time.time() - start_time < capture_duration and self.capture_active):
                # Read data from STM32N6 with timeout check
                current_time = time.time()
                data = self.serial.read(4096)
                
                if data:
                    self.buffer += data
                    last_frame_time = current_time
                    no_data_count = 0
                else:
                    no_data_count += 1
                    
                    # Check for frame timeout
                    if current_time - last_frame_time > self.frame_timeout:
                        print(f"⚠️ Warning: No frames received for {self.frame_timeout} seconds")
                        print("   Check if STM32N6 is running and camera is active")
                        
                        # Ask user if they want to continue
                        print("   Continue waiting? (y/n): ", end="", flush=True)
                        
                        # Non-blocking input check
                        import select
                        if select.select([sys.stdin], [], [], 0)[0]:
                            choice = sys.stdin.readline().strip().lower()
                            if choice.startswith('n'):
                                print("❌ Capture cancelled by user")
                                return False
                        
                        last_frame_time = current_time  # Reset timeout
                    
                    # Check for total connection loss
                    if no_data_count >= max_no_data:
                        print(f"❌ No data received for {max_no_data} consecutive reads")
                        print("   STM32N6 may have disconnected or stopped transmitting")
                        return False
                
                # Process ALN frames
                pos = 0
                while pos < len(self.buffer) and self.capture_active:
                    frame_data, next_pos = self.find_aln_frame(self.buffer, pos)
                    
                    if frame_data:
                        image = frame_data['image']
                        frames_processed += 1
                        
                        elapsed = time.time() - start_time
                        remaining = max(0, capture_duration - elapsed)
                        
                        print(f"📸 Frame {frames_processed} ({remaining:.1f}s left)...", end=" ")
                        
                        # Analyze quality
                        features = self.analyze_face_quality(image)
                        print(f"Quality: {features['quality']} ({features['quality_score']}/100)", end="")
                        
                        if features['quality_reasons']:
                            print(f" - Issues: {', '.join(features['quality_reasons'])}")
                        else:
                            print(" ✓")
                        
                        # Extract embedding if quality is good
                        embedding = None
                        if features['suitable_for_recognition'] and self.embedding_session:
                            try:
                                # Add timeout for embedding extraction
                                embedding_start = time.time()
                                embedding = self.extract_face_embedding(image)
                                embedding_time = time.time() - embedding_start
                                
                                if embedding is not None:
                                    print(f"   🧠 Extracted embedding (size: {len(embedding)}, time: {embedding_time:.2f}s)")
                                else:
                                    print("   ⚠️ Failed to extract embedding")
                                    
                            except Exception as e:
                                print(f"   ❌ Embedding extraction error: {e}")
                                embedding = None
                        
                        # Save face data
                        face_info = {
                            'face_id': frames_processed,
                            'quality': features['quality'],
                            'quality_score': features['quality_score'],
                            'timestamp': frame_data['timestamp']
                        }
                        
                        try:
                            saved_name, face_path, metadata_path = self.save_face_data(
                                image, features, embedding, frames_processed
                            )
                        except Exception as e:
                            print(f"   ⚠️ Failed to save face data: {e}")
                            saved_name = f"unsaved_frame_{frames_processed}"
                        
                        # Keep track of suitable faces for target
                        if (features['quality_score'] >= min_quality_score and 
                            embedding is not None):
                            suitable_faces.append({
                                'embedding': embedding,
                                'features': features,
                                'face_info': face_info,
                                'saved_name': saved_name
                            })
                            print(f"   ⭐ Added as target candidate (total: {len(suitable_faces)})")
                        
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
            
        except KeyboardInterrupt:
            print(f"\n⏹️ Capture interrupted by user")
            self.capture_active = False
            elapsed = time.time() - start_time
        except Exception as e:
            print(f"\n❌ Capture error: {e}")
            self.capture_active = False
            return False
        
        print(f"\n📊 Capture Summary:")
        print(f"   ⏱️  Duration: {elapsed:.1f} seconds")
        print(f"   📸 Frames processed: {frames_processed}")
        print(f"   ⭐ Suitable target faces: {len(suitable_faces)}")
        
        # Select best face for target
        if suitable_faces:
            # Sort by quality score and select the best
            best_face = max(suitable_faces, key=lambda x: x['features']['quality_score'])
            
            print(f"\n🎯 Selected best face for target:")
            print(f"   📁 File: {best_face['saved_name']}")
            print(f"   ⭐ Quality: {best_face['features']['quality']} ({best_face['features']['quality_score']}/100)")
            print(f"   🧠 Embedding size: {len(best_face['embedding'])}")
            
            # Update target embedding
            if self.update_target_embedding(best_face['embedding'], best_face['face_info']):
                print(f"\n✅ SUCCESS: Target embedding updated!")
                print(f"   📂 Updated file: {self.target_embedding_path}")
                print(f"   🔄 Rebuild your STM32N6 firmware to use the new target")
                print(f"   🎯 Your STM32N6 will now recognize this face!")
                return True
            else:
                print(f"\n❌ Failed to update target embedding file")
                return False
        else:
            print(f"\n⚠️  No suitable faces captured for target embedding")
            
            if frames_processed == 0:
                print(f"   💡 No frames received - check STM32N6 connection and camera")
            else:
                print(f"   💡 Try again with:")
                print(f"      • Better lighting")
                print(f"      • Stay still and look directly at camera")
                print(f"      • Lower --min-quality threshold")
                print(f"      • Longer --duration")
            return False
    
    def list_captured_faces(self):
        """List all captured faces with their quality"""
        faces_dir = Path(self.output_dir) / "faces"
        if not faces_dir.exists():
            print("No faces captured yet")
            return
        
        metadata_files = list(faces_dir.glob("*_metadata.json"))
        if not metadata_files:
            print("No face metadata found")
            return
        
        print(f"📁 Captured Faces ({len(metadata_files)} total):")
        print()
        
        for metadata_file in sorted(metadata_files):
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                
                face_id = metadata.get('face_id', 'unknown')
                quality = metadata['features'].get('quality', 'unknown')
                score = metadata['features'].get('quality_score', 0)
                timestamp = metadata.get('timestamp', 'unknown')
                has_embedding = metadata.get('embedding_available', False)
                
                status = "🧠" if has_embedding else "📸"
                print(f"   {status} Face {face_id:03d}: {quality} ({score}/100) - {timestamp}")
                
            except Exception as e:
                print(f"   ❌ Error reading {metadata_file}: {e}")
    
    def disconnect(self):
        """Disconnect from STM32N6"""
        self.capture_active = False
        if self.serial:
            try:
                self.serial.close()
                print("🔌 Disconnected from STM32N6")
            except Exception as e:
                print(f"⚠️ Error during disconnect: {e}")

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print(f"\n⏹️ Received interrupt signal - stopping capture...")
    # The capture loop will check capture_active flag
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(
        description="STM32N6 Face Recognition Target Updater",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture target face and update embedding
  python3 stm32_face_updater.py --capture-target
  
  # Capture for 60 seconds with lower quality threshold
  python3 stm32_face_updater.py --capture-target --duration 60 --min-quality 50
  
  # List all captured faces
  python3 stm32_face_updater.py --list-faces
  
  # Use different face recognition model
  python3 stm32_face_updater.py --capture-target --model custom_model.onnx
        """
    )
    
    parser.add_argument(
        "--capture-target", 
        action="store_true",
        help="Capture target face from STM32N6 and update embedding"
    )
    parser.add_argument(
        "--list-faces",
        action="store_true", 
        help="List all captured faces with quality scores"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Capture duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--min-quality",
        type=int,
        default=75,
        help="Minimum quality score for target embedding (default: 75)"
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyACM0",
        help="STM32N6 serial port (default: /dev/ttyACM0)"
    )
    parser.add_argument(
        "--model",
        default="input_models/mobilefacenet_int8_faces.onnx",
        help="Face recognition ONNX model path"
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=7372800,
        help="Serial baudrate (default: 7372800)"
    )
    parser.add_argument(
        "--connection-timeout",
        type=float,
        default=10.0,
        help="Connection timeout in seconds (default: 10.0)"
    )
    parser.add_argument(
        "--frame-timeout", 
        type=float,
        default=5.0,
        help="Frame timeout in seconds (default: 5.0)"
    )
    
    args = parser.parse_args()
    
    if not args.capture_target and not args.list_faces:
        parser.print_help()
        return
    
    # Create updater
    updater = STM32FaceRecognitionUpdater(args.port, args.baudrate)
    
    # Set custom timeouts if provided
    if hasattr(args, 'connection_timeout'):
        updater.connection_timeout = args.connection_timeout
    if hasattr(args, 'frame_timeout'):
        updater.frame_timeout = args.frame_timeout
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # List faces mode
    if args.list_faces:
        updater.list_captured_faces()
        return
    
    # Capture target mode
    if args.capture_target:
        print("🚀 STM32N6 Face Recognition Target Updater")
        print("=" * 50)
        
        # Connect to STM32N6
        if not updater.connect():
            print("❌ Failed to connect to STM32N6")
            return 1
        
        # Load face recognition model
        if not updater.load_face_recognition_model(args.model):
            print("❌ Failed to load face recognition model")
            updater.disconnect()
            return 1
        
        try:
            # Capture and update target
            success = updater.capture_and_update_target(
                capture_duration=args.duration,
                min_quality_score=args.min_quality
            )
            
            if success:
                print(f"\n🎉 Target embedding update completed successfully!")
                return 0
            else:
                print(f"\n❌ Target embedding update failed")
                return 1
                
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
            return 0
        finally:
            updater.disconnect()

if __name__ == "__main__":
    sys.exit(main())
