#!/usr/bin/env python3
"""
ALN Frame UI - Modified robust_ui to handle ALN frames from STM32N6
Designed to parse ALN camera frames instead of binary protocol messages
"""

import sys
import json
import time
import threading
import struct
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QTimer, Signal, QThread, QMutex, QMutexLocker
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout,
    QWidget, QLabel, QPushButton, QComboBox, QLineEdit, QProgressBar,
    QTextEdit, QTabWidget, QGroupBox, QCheckBox, QSpinBox, QSlider,
    QSplitter, QFrame, QScrollArea, QMessageBox, QFileDialog,
    QStatusBar, QMenuBar, QToolBar, QDialog
)
import serial
from serial.tools import list_ports

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ALNSettings:
    """ALN Frame UI settings"""
    baud_rate: int = 921600 * 8
    auto_reconnect: bool = True
    theme: str = "dark"
    save_frames: bool = True
    frame_save_path: str = "captured_frames"
    show_enhanced_view: bool = True
    analyze_face_quality: bool = True
    
    def save(self, path: Path):
        """Save settings to JSON file"""
        try:
            with open(path, 'w') as f:
                json.dump(asdict(self), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save settings: {e}")
    
    @classmethod
    def load(cls, path: Path) -> 'ALNSettings':
        """Load settings from JSON file"""
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                return cls(**data)
            except Exception as e:
                logger.warning(f"Failed to load settings: {e}")
        return cls()

class ALNImageWidget(QLabel):
    """Image display widget for ALN frames"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #444;
                border-radius: 8px;
                background-color: #222;
                color: #fff;
            }
        """)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText("No Camera Feed")
        self.setScaledContents(False)
        
        # Statistics
        self.frames_received = 0
        self.last_frame_time = 0
        self.frame_rate = 0.0
        
    def set_image(self, image: np.ndarray, frame_info: Dict[str, Any] = None):
        """Set image to display with optional frame analysis"""
        if image is None:
            self.setText("No Camera Feed")
            return
            
        try:
            # Update statistics
            current_time = time.time()
            if self.last_frame_time > 0:
                interval = current_time - self.last_frame_time
                if interval > 0:
                    self.frame_rate = 0.9 * self.frame_rate + 0.1 * (1.0 / interval)
            self.last_frame_time = current_time
            self.frames_received += 1
            
            # Analyze image quality if frame_info provided
            display_image = image.copy()
            
            if frame_info and frame_info.get('analyze', False):
                # Add analysis overlays
                display_image = self._add_analysis_overlay(display_image, frame_info)
            
            # Convert grayscale to RGB if needed
            if len(display_image.shape) == 2:
                image_rgb = cv2.cvtColor(display_image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = display_image
                
            height, width = image_rgb.shape[:2]
            
            # Create QImage
            if len(image_rgb.shape) == 3:
                bytes_per_line = 3 * width
                q_image = QtGui.QImage(
                    image_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
                )
            else:
                bytes_per_line = width
                q_image = QtGui.QImage(
                    image_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8
                )
            
            # Scale to fit widget
            pixmap = QtGui.QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
            
        except Exception as e:
            logger.error(f"Failed to display image: {e}")
            self.setText("Image Error")
    
    def _add_analysis_overlay(self, image: np.ndarray, frame_info: Dict[str, Any]) -> np.ndarray:
        """Add analysis overlay to image"""
        try:
            # Convert to RGB for overlay
            if len(image.shape) == 2:
                overlay_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                overlay_image = image.copy()
            
            # Add brightness analysis
            brightness = np.mean(image)
            if brightness < 80:
                color = (255, 100, 100)  # Red for too dark
                text = "TOO DARK"
            elif brightness > 180:
                color = (255, 255, 100)  # Yellow for too bright
                text = "TOO BRIGHT"
            else:
                color = (100, 255, 100)  # Green for good
                text = "GOOD LIGHT"
            
            cv2.putText(overlay_image, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(overlay_image, f"Brightness: {brightness:.0f}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Add contrast analysis
            contrast = np.std(image)
            cv2.putText(overlay_image, f"Contrast: {contrast:.0f}", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Add sharpness analysis (Laplacian variance)
            laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
            if laplacian_var < 100:
                sharpness_text = "BLURRY"
                sharpness_color = (255, 100, 100)
            else:
                sharpness_text = "SHARP"
                sharpness_color = (100, 255, 100)
            
            cv2.putText(overlay_image, f"Sharpness: {sharpness_text}", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, sharpness_color, 1)
            
            # Add frame info
            if 'frame_id' in frame_info:
                cv2.putText(overlay_image, f"Frame: {frame_info['frame_id']}", (5, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            
            return overlay_image
            
        except Exception as e:
            logger.error(f"Error adding overlay: {e}")
            return image

class ALNStatsWidget(QWidget):
    """Widget to show ALN frame statistics"""
    
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        # Connection status
        self.status_label = QLabel("Disconnected")
        self.port_label = QLabel("Port: None")
        
        # Frame statistics
        self.frame_count_label = QLabel("Frames: 0")
        self.frame_rate_label = QLabel("FPS: 0.0")
        self.throughput_label = QLabel("Throughput: 0.0 KB/s")
        
        # ALN statistics
        self.aln_found_label = QLabel("ALN Patterns: 0")
        self.aa_found_label = QLabel("0xAA Patterns: 0")
        self.bytes_received_label = QLabel("Bytes: 0")
        
        # Error statistics
        self.parse_errors_label = QLabel("Parse Errors: 0")
        self.frame_errors_label = QLabel("Frame Errors: 0")
        
        # Quality analysis
        self.brightness_label = QLabel("Brightness: 0")
        self.contrast_label = QLabel("Contrast: 0")
        self.sharpness_label = QLabel("Sharpness: 0")
        
        # Face analysis
        self.face_quality_label = QLabel("Face Quality: Unknown")
        self.last_save_label = QLabel("Last Save: None")
        
        layout.addWidget(QLabel("Connection:"))
        layout.addWidget(self.status_label)
        layout.addWidget(self.port_label)
        layout.addWidget(QLabel(""))
        
        layout.addWidget(QLabel("Frame Stats:"))
        layout.addWidget(self.frame_count_label)
        layout.addWidget(self.frame_rate_label)
        layout.addWidget(self.throughput_label)
        layout.addWidget(QLabel(""))
        
        layout.addWidget(QLabel("Protocol Stats:"))
        layout.addWidget(self.aln_found_label)
        layout.addWidget(self.aa_found_label)
        layout.addWidget(self.bytes_received_label)
        layout.addWidget(QLabel(""))
        
        layout.addWidget(QLabel("Error Stats:"))
        layout.addWidget(self.parse_errors_label)
        layout.addWidget(self.frame_errors_label)
        layout.addWidget(QLabel(""))
        
        layout.addWidget(QLabel("Quality Analysis:"))
        layout.addWidget(self.brightness_label)
        layout.addWidget(self.contrast_label)
        layout.addWidget(self.sharpness_label)
        layout.addWidget(QLabel(""))
        
        layout.addWidget(QLabel("Face Analysis:"))
        layout.addWidget(self.face_quality_label)
        layout.addWidget(self.last_save_label)
        
        layout.addStretch()
        
    def update_connection(self, connected: bool, port: str = ""):
        """Update connection status"""
        if connected:
            self.status_label.setText("Connected")
            self.status_label.setStyleSheet("color: green;")
            self.port_label.setText(f"Port: {port}")
        else:
            self.status_label.setText("Disconnected")
            self.status_label.setStyleSheet("color: red;")
            self.port_label.setText("Port: None")
    
    def update_stats(self, stats: Dict[str, Any]):
        """Update statistics display"""
        self.frame_count_label.setText(f"Frames: {stats.get('frames_received', 0)}")
        self.frame_rate_label.setText(f"FPS: {stats.get('frame_rate', 0.0):.1f}")
        self.throughput_label.setText(f"Throughput: {stats.get('throughput_kbps', 0.0):.1f} KB/s")
        
        self.aln_found_label.setText(f"ALN Patterns: {stats.get('aln_patterns', 0)}")
        self.aa_found_label.setText(f"0xAA Patterns: {stats.get('aa_patterns', 0)}")
        self.bytes_received_label.setText(f"Bytes: {stats.get('bytes_received', 0):,}")
        
        self.parse_errors_label.setText(f"Parse Errors: {stats.get('parse_errors', 0)}")
        self.frame_errors_label.setText(f"Frame Errors: {stats.get('frame_errors', 0)}")
        
        # Quality stats
        self.brightness_label.setText(f"Brightness: {stats.get('brightness', 0):.0f}")
        self.contrast_label.setText(f"Contrast: {stats.get('contrast', 0):.0f}")
        self.sharpness_label.setText(f"Sharpness: {stats.get('sharpness', 0):.0f}")
        
        # Face quality
        quality = stats.get('face_quality', 'Unknown')
        self.face_quality_label.setText(f"Face Quality: {quality}")
        
        # Last save
        last_save = stats.get('last_save_time', 'None')
        self.last_save_label.setText(f"Last Save: {last_save}")

class ALNFrameReader(QThread):
    """Thread to read ALN frames from serial port"""
    
    frame_received = Signal(np.ndarray, dict)  # New frame image + analysis
    stats_updated = Signal(dict)         # Statistics update
    error_occurred = Signal(str)         # Error message
    
    def __init__(self, serial_port, settings):
        super().__init__()
        self.serial_port = serial_port
        self.settings = settings
        self._running = False
        
        # Create save directory if needed
        if self.settings.save_frames:
            import os
            os.makedirs(self.settings.frame_save_path, exist_ok=True)
        
        # Statistics
        self.stats = {
            'frames_received': 0,
            'frame_rate': 0.0,
            'throughput_kbps': 0.0,
            'aln_patterns': 0,
            'aa_patterns': 0,
            'bytes_received': 0,
            'parse_errors': 0,
            'frame_errors': 0,
            'brightness': 0,
            'contrast': 0,
            'sharpness': 0,
            'face_quality': 'Unknown',
            'last_save_time': 'None'
        }
        
        # Frame processing
        self.buffer = b''
        self.last_frame_time = 0
        self.last_stats_time = time.time()
        self.frame_counter = 0
        
    def run(self):
        """Main reading loop"""
        self._running = True
        logger.info("ALN frame reader started")
        
        while self._running:
            try:
                # Read data
                data = self.serial_port.read(4096)
                if data:
                    self.buffer += data
                    self.stats['bytes_received'] += len(data)
                    
                    # Count patterns
                    self.stats['aln_patterns'] += data.count(b'ALN')
                    self.stats['aa_patterns'] += data.count(b'\xaa')
                    
                    # Process ALN frames
                    self._process_aln_frames()
                    
                    # Update stats periodically
                    current_time = time.time()
                    if current_time - self.last_stats_time >= 1.0:
                        self._update_stats()
                        self.last_stats_time = current_time
                
                # Keep buffer size manageable
                if len(self.buffer) > 100000:  # 100KB
                    self.buffer = self.buffer[-50000:]  # Keep last 50KB
                
                self.msleep(10)  # 10ms sleep
                
            except Exception as e:
                logger.error(f"Error in frame reader: {e}")
                self.error_occurred.emit(str(e))
                self.msleep(100)
        
        logger.info("ALN frame reader stopped")
    
    def _process_aln_frames(self):
        """Process ALN frames in buffer"""
        pos = 0
        while pos < len(self.buffer):
            # Look for ALN pattern
            aln_pos = self.buffer.find(b'ALN', pos)
            if aln_pos == -1:
                break
            
            # Check if we have enough data for frame header + image
            frame_start = aln_pos
            header_end = aln_pos + 15  # ALN + header
            
            if header_end > len(self.buffer):
                # Not enough data for header
                break
            
            try:
                # Parse ALN frame structure
                # Expected: ALN 00 70 00 00 00 70 00 00 00 (112x112)
                header = self.buffer[aln_pos:header_end]
                
                if len(header) >= 15:
                    # Extract dimensions
                    width = header[4] if header[4] > 0 else 112  # Default to 112
                    height = header[9] if header[9] > 0 else 112  # Default to 112
                    
                    # Validate dimensions
                    if width > 0 and height > 0 and width <= 640 and height <= 480:
                        image_size = width * height
                        image_end = header_end + image_size
                        
                        if image_end <= len(self.buffer):
                            # Extract image data
                            image_data = self.buffer[header_end:image_end]
                            
                            # Convert to numpy array
                            image_array = np.frombuffer(image_data, dtype=np.uint8)
                            image_array = image_array.reshape((height, width))
                            
                            # Analyze frame quality
                            frame_info = self._analyze_frame(image_array)
                            frame_info['frame_id'] = self.frame_counter
                            frame_info['analyze'] = self.settings.analyze_face_quality
                            
                            # Save frame if enabled and quality is good
                            if self.settings.save_frames and frame_info.get('save_worthy', False):
                                self._save_frame(image_array, frame_info)
                            
                            # Emit the frame with analysis
                            self.frame_received.emit(image_array, frame_info)
                            self.stats['frames_received'] += 1
                            self.frame_counter += 1
                            
                            # Update stats with quality info
                            self.stats.update({
                                'brightness': frame_info.get('brightness', 0),
                                'contrast': frame_info.get('contrast', 0),
                                'sharpness': frame_info.get('sharpness', 0),
                                'face_quality': frame_info.get('quality_text', 'Unknown')
                            })
                            
                            # Update frame rate
                            current_time = time.time()
                            if self.last_frame_time > 0:
                                interval = current_time - self.last_frame_time
                                if interval > 0:
                                    new_fps = 1.0 / interval
                                    self.stats['frame_rate'] = 0.9 * self.stats['frame_rate'] + 0.1 * new_fps
                            self.last_frame_time = current_time
                            
                            # Move past this frame
                            pos = image_end
                            continue
                
            except Exception as e:
                logger.debug(f"Frame parsing error: {e}")
                self.stats['parse_errors'] += 1
            
            # Move past this ALN marker
            pos = aln_pos + 1
    
    def _update_stats(self):
        """Calculate and emit updated statistics"""
        current_time = time.time()
        time_diff = current_time - self.last_stats_time
        
        if time_diff > 0:
            # Calculate throughput (last second)
            bytes_per_sec = self.stats['bytes_received'] / time_diff if time_diff > 0 else 0
            self.stats['throughput_kbps'] = bytes_per_sec / 1024
        
        self.stats_updated.emit(self.stats.copy())
    
    def _analyze_frame(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze frame quality and characteristics"""
        try:
            analysis = {}
            
            # Basic image statistics
            analysis['brightness'] = float(np.mean(image))
            analysis['contrast'] = float(np.std(image))
            
            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
            analysis['sharpness'] = float(laplacian_var)
            
            # Quality assessment
            quality_score = 0
            quality_reasons = []
            
            # Brightness check (optimal range 80-180)
            if 80 <= analysis['brightness'] <= 180:
                quality_score += 25
            else:
                quality_reasons.append("Poor lighting")
            
            # Contrast check (minimum 30)
            if analysis['contrast'] >= 30:
                quality_score += 25
            else:
                quality_reasons.append("Low contrast")
            
            # Sharpness check (minimum 100)
            if analysis['sharpness'] >= 100:
                quality_score += 25
            else:
                quality_reasons.append("Blurry")
            
            # Face coverage check (ensure face fills most of the frame)
            non_zero_pixels = np.count_nonzero(image > 10)  # Ignore very dark pixels
            total_pixels = image.shape[0] * image.shape[1]
            coverage = non_zero_pixels / total_pixels
            
            if coverage >= 0.3:  # At least 30% of frame has content
                quality_score += 25
            else:
                quality_reasons.append("Poor face coverage")
            
            analysis['coverage'] = coverage
            analysis['quality_score'] = quality_score
            
            # Quality classification
            if quality_score >= 75:
                analysis['quality_text'] = "EXCELLENT"
                analysis['quality_color'] = "green"
                analysis['save_worthy'] = True
            elif quality_score >= 50:
                analysis['quality_text'] = "GOOD"
                analysis['quality_color'] = "yellow"
                analysis['save_worthy'] = True
            elif quality_score >= 25:
                analysis['quality_text'] = "FAIR"
                analysis['quality_color'] = "orange"
                analysis['save_worthy'] = False
            else:
                analysis['quality_text'] = "POOR"
                analysis['quality_color'] = "red"
                analysis['save_worthy'] = False
            
            if quality_reasons:
                analysis['quality_text'] += f" ({', '.join(quality_reasons)})"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return {'quality_text': 'ERROR', 'save_worthy': False}
    
    def _save_frame(self, image: np.ndarray, frame_info: Dict[str, Any]):
        """Save frame to disk with metadata"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quality = frame_info.get('quality_score', 0)
            filename = f"face_{timestamp}_q{quality:02d}_f{frame_info.get('frame_id', 0):04d}.png"
            filepath = Path(self.settings.frame_save_path) / filename
            
            # Save image
            cv2.imwrite(str(filepath), image)
            
            # Save metadata
            metadata_file = filepath.with_suffix('.json')
            metadata = {
                'timestamp': timestamp,
                'frame_id': frame_info.get('frame_id', 0),
                'quality_score': quality,
                'brightness': frame_info.get('brightness', 0),
                'contrast': frame_info.get('contrast', 0),
                'sharpness': frame_info.get('sharpness', 0),
                'coverage': frame_info.get('coverage', 0),
                'quality_text': frame_info.get('quality_text', 'Unknown'),
                'dimensions': f"{image.shape[1]}x{image.shape[0]}"
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.stats['last_save_time'] = timestamp
            logger.info(f"Saved frame: {filename} (Quality: {quality})")
            
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
    
    def stop(self):
        """Stop the reader thread"""
        self._running = False

class ALNMainWindow(QMainWindow):
    """Main window for ALN frame visualization"""
    
    def __init__(self):
        super().__init__()
        self.settings = ALNSettings.load(Path("aln_settings.json"))
        self.serial_port: Optional[serial.Serial] = None
        self.frame_reader: Optional[ALNFrameReader] = None
        
        self.setWindowTitle("STM32N6 ALN Frame Viewer")
        self.setMinimumSize(1200, 800)
        
        self.setup_ui()
        self.apply_theme()
        self.refresh_ports()
        
    def setup_ui(self):
        """Setup user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - controls and stats
        left_panel = QWidget()
        left_panel.setFixedWidth(280)
        left_layout = QVBoxLayout(left_panel)
        
        # Connection controls
        conn_group = QGroupBox("Connection")
        conn_layout = QVBoxLayout(conn_group)
        
        # Port selection
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        port_layout.addWidget(self.port_combo)
        
        self.refresh_btn = QPushButton("🔄")
        self.refresh_btn.setFixedWidth(30)
        self.refresh_btn.clicked.connect(self.refresh_ports)
        port_layout.addWidget(self.refresh_btn)
        
        conn_layout.addLayout(port_layout)
        
        # Connect/Disconnect button
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)
        conn_layout.addWidget(self.connect_btn)
        
        left_layout.addWidget(conn_group)
        
        # Statistics widget
        self.stats_widget = ALNStatsWidget()
        left_layout.addWidget(self.stats_widget)
        
        # Log widget
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_widget = QTextEdit()
        self.log_widget.setMaximumHeight(150)
        self.log_widget.setStyleSheet("font-family: monospace; font-size: 10px;")
        log_layout.addWidget(self.log_widget)
        
        left_layout.addWidget(log_group)
        
        # Main image display
        self.image_widget = ALNImageWidget()
        
        # Add to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.image_widget, 1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
    def apply_theme(self):
        """Apply dark theme"""
        if self.settings.theme == "dark":
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QWidget {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QGroupBox {
                    font-weight: bold;
                    border: 2px solid #555;
                    border-radius: 5px;
                    margin-top: 1ex;
                    padding-top: 5px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
                QPushButton {
                    background-color: #4CAF50;
                    border: none;
                    color: white;
                    padding: 6px 12px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
                QComboBox {
                    border: 1px solid #555;
                    border-radius: 3px;
                    padding: 3px;
                    background-color: #444;
                }
                QTextEdit {
                    border: 1px solid #555;
                    border-radius: 3px;
                    background-color: #333;
                }
            """)
    
    def refresh_ports(self):
        """Refresh available serial ports"""
        self.port_combo.clear()
        ports = list_ports.comports()
        
        # Add STM32/ST-Link ports first
        stm_ports = []
        other_ports = []
        
        for port in ports:
            port_info = f"{port.device}"
            if port.description:
                port_info += f" - {port.description}"
            
            if ('STM' in str(port.manufacturer) or 
                'STM' in str(port.description) or
                'ACM' in port.device):
                stm_ports.append((port.device, port_info))
            else:
                other_ports.append((port.device, port_info))
        
        # Add STM ports first
        for device, info in stm_ports:
            self.port_combo.addItem(info, device)
        
        # Add other ports
        for device, info in other_ports:
            self.port_combo.addItem(info, device)
        
        if stm_ports:
            self.port_combo.setCurrentIndex(0)  # Select first STM port
        
        self.log_message(f"Found {len(ports)} serial ports")
    
    def toggle_connection(self):
        """Connect or disconnect from serial port"""
        if self.serial_port and self.serial_port.is_open:
            self.disconnect()
        else:
            self.connect()
    
    def connect(self):
        """Connect to selected serial port"""
        if self.port_combo.currentData():
            port = self.port_combo.currentData()
            try:
                self.serial_port = serial.Serial(
                    port, 
                    self.settings.baud_rate, 
                    timeout=0.1
                )
                
                # Start frame reader
                self.frame_reader = ALNFrameReader(self.serial_port, self.settings)
                self.frame_reader.frame_received.connect(self.on_frame_received)
                self.frame_reader.stats_updated.connect(self.on_stats_updated)
                self.frame_reader.error_occurred.connect(self.on_error)
                self.frame_reader.start()
                
                self.connect_btn.setText("Disconnect")
                self.stats_widget.update_connection(True, port)
                self.log_message(f"Connected to {port}")
                self.status_bar.showMessage(f"Connected to {port}")
                
            except Exception as e:
                self.log_message(f"Connection failed: {e}")
                QMessageBox.warning(self, "Connection Error", f"Failed to connect to {port}:\\n{e}")
    
    def disconnect(self):
        """Disconnect from serial port"""
        if self.frame_reader:
            self.frame_reader.stop()
            self.frame_reader.wait(2000)  # Wait up to 2 seconds
            self.frame_reader = None
        
        if self.serial_port:
            self.serial_port.close()
            self.serial_port = None
        
        self.connect_btn.setText("Connect")
        self.stats_widget.update_connection(False)
        self.log_message("Disconnected")
        self.status_bar.showMessage("Disconnected")
    
    def on_frame_received(self, image: np.ndarray, frame_info: Dict[str, Any]):
        """Handle new frame with analysis"""
        self.image_widget.set_image(image, frame_info)
    
    def on_stats_updated(self, stats: Dict[str, Any]):
        """Handle stats update"""
        self.stats_widget.update_stats(stats)
    
    def on_error(self, error: str):
        """Handle error"""
        self.log_message(f"Error: {error}")
    
    def log_message(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_widget.append(f"[{timestamp}] {message}")
        
        # Limit log lines
        document = self.log_widget.document()
        if document.blockCount() > 100:
            cursor = self.log_widget.textCursor()
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.Start)
            cursor.select(QtGui.QTextCursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
    
    def closeEvent(self, event):
        """Handle window close"""
        self.disconnect()
        self.settings.save(Path("aln_settings.json"))
        event.accept()

def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("STM32N6 ALN Frame Viewer")
    
    window = ALNMainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
