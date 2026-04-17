#!/usr/bin/env python3
"""
CSV to ROS BAG Converter for Magnetometer Calibration Data
Converts magnetometer CSV files to ROS BAG format
"""

import os
import sys
import csv
from pathlib import Path
from datetime import datetime

try:
    import rosbag
    from genpy.rostime import Time as RospyTime
    from genpy import Message
    import struct
    
except ImportError as e:
    print(f"ERROR: ROS not properly installed or rosbag not available: {e}")
    print("\nTo fix this, you need to:")
    print("1. Make sure you have ROS installed")
    print("2. Source your ROS setup: source /opt/ros/<distro>/setup.bash")
    print("3. Or: pip install rosbag")
    sys.exit(1)


class CSVToBagConverter:
    """Converts CSV calibration data to ROS BAG files"""
    
    def __init__(self, input_dir, output_dir):
        """
        Initialize the converter
        
        Args:
            input_dir: Directory containing CSV files
            output_dir: Directory to save BAG files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def parse_csv_file(self, csv_path):
        """
        Parse a CSV file and extract magnetometer data and calibration parameters
        
        Returns:
            tuple: (data_points, calibration_params)
                - data_points: list of {'X': x, 'Y': y, 'Z': z}
                - calibration_params: dict with HI_OFFSET, SI_DISTORTION, SI_ROTATION
        """
        calibration_params = {}
        data_points = []
        
        # utf-8-sig entfernt ein moegliches BOM am Dateianfang (\ufeff)
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        
        # Parse header lines starting with '#'
        data_start_idx = 0
        for idx, line in enumerate(lines):
            line = line.strip()
            if line.startswith('#'):
                # Parse calibration header
                if ':' in line:
                    key, value = line[1:].split(':', 1)
                    calibration_params[key.strip()] = value.strip()
            else:
                # First non-comment line should be the CSV header
                data_start_idx = idx
                break
        
        # Parse CSV data from data_start_idx onwards
        csv_text = ''.join(lines[data_start_idx:])
        reader = csv.DictReader(csv_text.splitlines(), delimiter=';')
        
        if reader.fieldnames is None:
            print(f"ERROR: Could not read CSV headers from {csv_path}")
            return None, None
        
        for row in reader:
            try:
                # Normalize keys/values and strip BOM artifacts from header names.
                clean_row = {
                    str(k).replace('\ufeff', '').strip(): str(v).strip()
                    for k, v in row.items()
                    if k is not None
                }
                data_points.append({
                    'X': float(clean_row['X']),
                    'Y': float(clean_row['Y']),
                    'Z': float(clean_row['Z'])
                })
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not parse row {row}: {e}")
                continue
        
        return data_points, calibration_params
    
    def create_magnetic_field_message(self, x, y, z, seq=0):
        """
        Create a simple magnetometer message compatible with sensor_msgs_ext/magnetometer
        
        Args:
            x, y, z: Magnetic field components
            seq: Sequence number (not used for custom message)
            
        Returns:
            A message-like object with x, y, z fields
        """
        # Create a simple message-like object with proper ROS message structure
        class MagnetometerMessage:
            _type = 'sensor_msgs_ext/magnetometer'
            _md5sum = '4a842b65f413084dc2b10fb484ea7f17'  # Correct MD5 for float64 x,y,z
            _full_text = '''float64 x
float64 y
float64 z
'''
            __slots__ = ('x', 'y', 'z')
            
            def __init__(self):
                self.x = 0.0
                self.y = 0.0
                self.z = 0.0
            
            def serialize(self, buff):
                """Serialize message to buffer"""
                buff.write(struct.pack('<d', self.x))
                buff.write(struct.pack('<d', self.y))
                buff.write(struct.pack('<d', self.z))
        
        msg = MagnetometerMessage()
        msg.x = x
        msg.y = y
        msg.z = z
        
        return msg
    
    def csv_to_bag(self, csv_file, bag_file=None):
        """
        Convert a CSV file to a BAG file
        
        Args:
            csv_file: Path to CSV file
            bag_file: Path to output BAG file (optional, auto-generated if None)
            
        Returns:
            bool: True if successful, False otherwise
        """
        csv_path = self.input_dir / csv_file
        
        if not csv_path.exists():
            print(f"ERROR: CSV file not found: {csv_path}")
            return False
        
        # Parse CSV
        print(f"\nProcessing: {csv_file}")
        data_points, calibration_params = self.parse_csv_file(csv_path)
        
        if data_points is None:
            print(f"ERROR: Failed to parse {csv_file}")
            return False
        
        print(f"  - Found {len(data_points)} data points")
        print(f"  - Calibration params: {calibration_params}")
        
        # Generate output filename if not provided
        if bag_file is None:
            base_name = csv_path.stem
            bag_file = f"{base_name}.bag"
        
        bag_path = self.output_dir / bag_file
        
        # Create BAG file
        try:
            with rosbag.Bag(str(bag_path), 'w') as bag:
                # Start with a reasonable base timestamp (e.g., Jan 01 2020)
                # This avoids "out of dual 32-bit range" errors
                base_timestamp = RospyTime.from_sec(1577836800)  # Jan 1, 2020
                
                for idx, data_point in enumerate(data_points):
                    # Create timestamp for each point (increment by 0.01 seconds)
                    # Add to base timestamp instead of starting from 0
                    timestamp = RospyTime.from_sec(base_timestamp.to_sec() + idx * 0.01)
                    
                    # Create ROS message (custom sensor_msgs_ext/magnetometer)
                    msg = self.create_magnetic_field_message(
                        data_point['X'],
                        data_point['Y'],
                        data_point['Z'],
                        seq=idx
                    )
                    
                    # Write to BAG file
                    # Use topic name that matches ROS convention
                    bag.write('/imu/magnetometer', msg, timestamp)
                
                # Log calibration parameters as a message for reference
                print(f"  - Wrote {len(data_points)} messages to {bag_path}")
            
            print(f"  ✓ Successfully created: {bag_file}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to create BAG file: {e}")
            return False
    
    def convert_all(self):
        """
        Convert all CSV files in input directory to BAG files
        
        Returns:
            dict: Statistics about conversion
        """
        csv_files = sorted(self.input_dir.glob('*.csv'))
        
        if not csv_files:
            print(f"ERROR: No CSV files found in {self.input_dir}")
            return None
        
        print(f"\n{'='*60}")
        print(f"CSV to BAG Converter")
        print(f"{'='*60}")
        print(f"Input directory:  {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Found {len(csv_files)} CSV file(s)")
        
        stats = {
            'total': len(csv_files),
            'successful': 0,
            'failed': 0
        }
        
        # Prüfe, ob BAG-Datei schon existiert
        for csv_file in csv_files:
            bag_name = csv_file.with_suffix('.bag').name
            bag_path = self.output_dir / bag_name
            if bag_path.exists():
                print(f"  - Überspringe {csv_file.name}: BAG existiert bereits.")
                continue
            if self.csv_to_bag(csv_file.name):
                stats['successful'] += 1
            else:
                stats['failed'] += 1
        
        print(f"\n{'='*60}")
        print(f"Conversion Summary:")
        print(f"  Total:      {stats['total']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Failed:     {stats['failed']}")
        print(f"{'='*60}\n")
        
        return stats


def main():
    """Main function"""
    
    # Define paths
    script_dir = Path(__file__).parent
    
    # Allow command line arguments for flexibility
    if len(sys.argv) > 1:
        input_dir = Path(sys.argv[1])
    else:
        input_dir = script_dir /  "1-kalipoints_exports/neu"
    
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = script_dir / "2-kali-exports_transformation_for_calibration/neu"
    
    # Create converter
    converter = CSVToBagConverter(input_dir, output_dir)
    
    # Convert all CSV files
    stats = converter.convert_all()
    
    if stats and stats['successful'] > 0:
        print(f"✓ All conversions completed successfully!")
        print(f"\nBAG files are saved in: {output_dir}")
        return 0
    else:
        print(f"✗ Conversion failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
