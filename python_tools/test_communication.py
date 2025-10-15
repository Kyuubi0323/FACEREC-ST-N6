#!/usr/bin/env python3
"""
Simple command-line tool to test STM32N6 communication using robust_protocol
"""

import serial
import time
import sys
from robust_protocol import RobustProtocolParser, MessageType

def test_stm32_communication():
    """Test communication with STM32N6 using robust protocol parser"""
    
    # Try to find ST-Link VCP port
    try:
        from serial.tools import list_ports
        ports = list_ports.comports()
        stlink_port = None
        
        for port in ports:
            if 'STMicroelectronics' in str(port.manufacturer) or 'STM' in str(port.description):
                stlink_port = port.device
                break
        
        if not stlink_port:
            # Fall back to common STM32 VCP port
            stlink_port = '/dev/ttyACM0'
        
        print(f"🔗 Attempting to connect to {stlink_port}")
        
    except Exception as e:
        print(f"⚠️ Error detecting port: {e}")
        stlink_port = '/dev/ttyACM0'
    
    try:
        # Connect to STM32N6
        ser = serial.Serial(stlink_port, 7372800, timeout=0.1)
        print(f"✅ Connected to {stlink_port} at 7.37 Mbps")
        
        # Initialize protocol parser
        parser = RobustProtocolParser()
        
        # Statistics
        bytes_received = 0
        messages_parsed = 0
        frames_found = 0
        aln_patterns = 0
        aa_patterns = 0
        
        print("🔍 Monitoring STM32N6 data stream...")
        print("Press Ctrl+C to stop\n")
        
        start_time = time.time()
        
        while True:
            # Read data
            data = ser.read(4096)
            if data:
                bytes_received += len(data)
                
                # Count patterns
                aln_patterns += data.count(b'ALN')
                aa_patterns += data.count(b'\xaa')
                
                # Try to parse with robust protocol
                try:
                    messages = parser.process_data(data)
                    messages_parsed += len(messages)
                    
                    for msg in messages:
                        if msg.message_type == MessageType.FRAME_DATA:
                            frames_found += 1
                            print(f"📸 Frame data message: {len(msg.payload)} bytes")
                        elif msg.message_type == MessageType.DETECTION_RESULTS:
                            print(f"🎯 Detection results: {len(msg.payload)} bytes")
                        elif msg.message_type == MessageType.EMBEDDING_DATA:
                            print(f"🧠 Embedding data: {len(msg.payload)} bytes")
                        else:
                            print(f"📨 Message type {msg.message_type}: {len(msg.payload)} bytes")
                            
                except Exception as e:
                    # Robust protocol might fail, that's ok
                    pass
                
                # Show periodic stats
                elapsed = time.time() - start_time
                if elapsed >= 2.0:  # Every 2 seconds
                    throughput_kbps = (bytes_received / elapsed) / 1024
                    
                    print(f"\n📊 Stats (last {elapsed:.1f}s):")
                    print(f"   📦 Bytes received: {bytes_received:,}")
                    print(f"   🚀 Throughput: {throughput_kbps:.1f} KB/s")
                    print(f"   📨 Messages parsed: {messages_parsed}")
                    print(f"   🖼️  Frames found: {frames_found}")
                    print(f"   🔵 ALN patterns: {aln_patterns}")
                    print(f"   🟡 0xAA patterns: {aa_patterns}")
                    
                    # Get parser stats
                    stats = parser.get_stats()
                    if stats['bytes_received'] > 0:
                        print(f"   ⚠️  Sync errors: {stats['sync_errors']}")
                        print(f"   ❌ CRC errors: {stats['crc_errors']}")
                        print(f"   🔄 Messages dropped: {stats['messages_dropped']}")
                    
                    print("   (Press Ctrl+C to stop)\n")
                    
                    # Reset for next period
                    bytes_received = 0
                    messages_parsed = 0
                    frames_found = 0
                    aln_patterns = 0
                    aa_patterns = 0
                    start_time = time.time()
            
            time.sleep(0.01)  # Small delay
            
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("🔌 Disconnected")

if __name__ == "__main__":
    test_stm32_communication()
