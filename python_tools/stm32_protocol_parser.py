#!/usr/bin/env python3
"""
STM32N6 Face Recognition Protocol Parser
Decodes the mixed protocol data stream from STM32N6
"""

import serial
import struct
import time
import numpy as np
import cv2
from PIL import Image
import os

class STM32ProtocolParser:
    def __init__(self, port='/dev/ttyACM0', baudrate=7372800):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.buffer = b''
        self.frame_count = 0
        self.output_dir = "captured_frames"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def connect(self):
        """Connect to the STM32N6 board"""
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            print(f"✅ Connected to {self.port} at {self.baudrate} baud")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the STM32N6 board"""
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("🔌 Disconnected")
    
    def find_aln_frame(self, buffer, start_pos=0):
        """Find ALN frame in buffer and extract it"""
        # Look for ALN pattern
        aln_pos = buffer.find(b'ALN', start_pos)
        if aln_pos == -1:
            return None, len(buffer)
        
        # Check if we have enough data for the header
        if aln_pos + 20 > len(buffer):
            return None, aln_pos
        
        try:
            # Extract ALN frame header (based on observed pattern)
            # ALN + width(2) + height(2) + unknown(4) + data
            header = buffer[aln_pos:aln_pos + 15]
            
            # Parse header - ALN followed by dimensions
            if len(header) >= 15:
                # Format appears to be: ALN 00 70 00 00 00 70 00 00 00
                # This suggests 112x112 (0x70 = 112)
                width = 112
                height = 112
                data_size = width * height  # Grayscale image
                
                frame_start = aln_pos + 15
                frame_end = frame_start + data_size
                
                # Check if we have the complete frame
                if frame_end <= len(buffer):
                    frame_data = buffer[frame_start:frame_end]
                    return {
                        'type': 'ALN_FRAME',
                        'width': width,
                        'height': height,
                        'data': frame_data,
                        'header': header,
                        'position': aln_pos
                    }, frame_end
                else:
                    # Not enough data yet
                    return None, aln_pos
        
        except Exception as e:
            print(f"⚠️ Error parsing ALN frame: {e}")
            return None, aln_pos + 1
        
        return None, aln_pos + 1
    
    def find_aa_message(self, buffer, start_pos=0):
        """Find 0xAA message in buffer"""
        aa_pos = buffer.find(b'\xaa', start_pos)
        if aa_pos == -1:
            return None, len(buffer)
        
        # Check if this 0xAA is part of an ALN frame
        # Look ahead for ALN pattern
        search_end = min(aa_pos + 20, len(buffer))
        aln_check = buffer[aa_pos:search_end]
        if b'ALN' in aln_check:
            # This 0xAA is part of an ALN frame, skip it
            return None, aa_pos + 1
        
        # Try to parse as a binary message
        if aa_pos + 10 > len(buffer):
            return None, aa_pos
        
        try:
            # Extract potential message header
            header = buffer[aa_pos:aa_pos + 10]
            return {
                'type': 'AA_MESSAGE',
                'data': header,
                'position': aa_pos
            }, aa_pos + 10
        
        except Exception as e:
            return None, aa_pos + 1
    
    def save_frame(self, frame_data, width, height, frame_num):
        """Save frame as image file"""
        try:
            # Convert to numpy array
            img_array = np.frombuffer(frame_data, dtype=np.uint8)
            img_array = img_array.reshape((height, width))
            
            # Save as PNG
            filename = f"{self.output_dir}/frame_{frame_num:04d}.png"
            cv2.imwrite(filename, img_array)
            
            # Also save as JPG for better compression
            jpg_filename = f"{self.output_dir}/frame_{frame_num:04d}.jpg"
            cv2.imwrite(jpg_filename, img_array)
            
            return filename
        
        except Exception as e:
            print(f"⚠️ Error saving frame {frame_num}: {e}")
            return None
    
    def process_data(self, max_frames=10, duration=30):
        """Process incoming data and extract frames"""
        if not self.serial or not self.serial.is_open:
            print("❌ Not connected to STM32N6")
            return
        
        print(f"🔍 Processing data for {duration} seconds or {max_frames} frames...")
        start_time = time.time()
        frames_captured = 0
        bytes_processed = 0
        
        while (time.time() - start_time < duration and 
               frames_captured < max_frames):
            
            # Read new data
            new_data = self.serial.read(4096)
            if new_data:
                self.buffer += new_data
                bytes_processed += len(new_data)
            
            # Process buffer for ALN frames
            pos = 0
            while pos < len(self.buffer):
                frame, next_pos = self.find_aln_frame(self.buffer, pos)
                
                if frame:
                    frames_captured += 1
                    print(f"📸 Frame {frames_captured}: {frame['width']}x{frame['height']} at position {frame['position']}")
                    
                    # Save the frame
                    filename = self.save_frame(
                        frame['data'], 
                        frame['width'], 
                        frame['height'], 
                        frames_captured
                    )
                    
                    if filename:
                        print(f"   💾 Saved as {filename}")
                    
                    # Show hex dump of header
                    hex_header = ' '.join(f'{b:02x}' for b in frame['header'])
                    print(f"   📋 Header: {hex_header}")
                    
                    pos = next_pos
                else:
                    pos = next_pos
                    if pos >= len(self.buffer):
                        break
            
            # Keep only last 64KB in buffer to prevent memory issues
            if len(self.buffer) > 65536:
                self.buffer = self.buffer[-32768:]
                pos = 0
            
            time.sleep(0.01)  # Small delay to prevent CPU overload
        
        elapsed = time.time() - start_time
        print(f"\n📊 Processing Summary:")
        print(f"   ⏱️  Duration: {elapsed:.1f} seconds")
        print(f"   📸 Frames captured: {frames_captured}")
        print(f"   📦 Bytes processed: {bytes_processed:,}")
        print(f"   🚀 Throughput: {bytes_processed/elapsed/1024:.1f} KB/s")
        
        if frames_captured > 0:
            print(f"   📁 Frames saved to: {os.path.abspath(self.output_dir)}")

def main():
    """Main function for testing"""
    parser = STM32ProtocolParser()
    
    if parser.connect():
        try:
            parser.process_data(max_frames=5, duration=20)
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        finally:
            parser.disconnect()
    else:
        print("❌ Failed to connect to STM32N6")

if __name__ == "__main__":
    main()
