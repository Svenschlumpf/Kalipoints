#!/usr/bin/env python3
"""
ROS2 BAG (.db3) to CSV Converter for Magnetometer Data
Reads ROS2 bag files directly via SQLite3 and manually parses CDR messages.
No ROS2 installation required - runs on Windows directly.

Message type: magnetometer_sensor_interfaces_rm3100iff/msg/Magnetometer
  std_msgs/Header header
    builtin_interfaces/Time stamp (int32 sec, uint32 nanosec)
    string frame_id
  geometry_msgs/Vector3 magnetic_field (float64 x, y, z)
  float64[9] magnetic_field_covariance
  std_msgs/Bool is_valid
  diagnostic_msgs/DiagnosticStatus diagnostic_status (skipped)
"""

import sqlite3
import struct
import csv
import sys
from pathlib import Path


TOPIC_NAME = '/nanoauv/gnc/navigation_sensors/magnetometer/magnetic_field'
CDR_HEADER_SIZE = 4  # 4-byte CDR encapsulation header


def align(offset: int, alignment: int) -> int:
    """Advance offset to the next multiple of alignment.
    
    IMPORTANT: CDR alignment is computed relative to the start of the CDR
    data stream (i.e. AFTER the 4-byte encapsulation header), not from
    the start of the raw blob.
    """
    stream_pos = offset - CDR_HEADER_SIZE
    remainder = stream_pos % alignment
    if remainder != 0:
        stream_pos += alignment - remainder
    return stream_pos + CDR_HEADER_SIZE


def parse_cdr_magnetometer(data: bytes):
    """
    Parse a CDR-serialized Magnetometer message blob from a ROS2 bag.

    CDR encapsulation header (4 bytes):
      byte[0]: 0x00
      byte[1]: 0x01 = little-endian, 0x00 = big-endian
      byte[2-3]: 0x00 0x00 (options)

    Returns a dict with: sec, nanosec, frame_id, x, y, z, is_valid
    Returns None on parse error.
    """
    if len(data) < 4:
        return None

    endian = '<' if data[1] == 0x01 else '>'
    offset = 4  # skip 4-byte CDR encapsulation header

    try:
        # ── std_msgs/Header ─────────────────────────────────────
        # stamp.sec  (int32, 4-byte aligned)
        offset = align(offset, 4)
        sec = struct.unpack_from(f'{endian}i', data, offset)[0]
        offset += 4

        # stamp.nanosec  (uint32, 4-byte aligned)
        offset = align(offset, 4)
        nanosec = struct.unpack_from(f'{endian}I', data, offset)[0]
        offset += 4

        # frame_id  (string: uint32 length + bytes incl. null terminator)
        offset = align(offset, 4)
        str_len = struct.unpack_from(f'{endian}I', data, offset)[0]
        offset += 4
        frame_id = data[offset:offset + str_len - 1].decode('utf-8') if str_len > 0 else ''
        offset += str_len

        # ── geometry_msgs/Vector3  magnetic_field ────────────────
        # float64 needs 8-byte alignment
        offset = align(offset, 8)
        x, y, z = struct.unpack_from(f'{endian}3d', data, offset)
        offset += 24  # 3 × 8 bytes

        # ── float64[9]  magnetic_field_covariance (skipped) ──────
        offset += 72  # 9 × 8 bytes

        # ── std_msgs/Bool  is_valid (1 byte, no alignment needed) ─
        is_valid = bool(data[offset]) if offset < len(data) else None

        return {
            'sec': sec,
            'nanosec': nanosec,
            'frame_id': frame_id,
            'x': x,
            'y': y,
            'z': z,
            'is_valid': is_valid,
        }

    except struct.error as e:
        return None


def bag_to_csv(bag_folder: Path, output_folder: Path) -> bool:
    """Convert a single ROS2 bag folder to a CSV file. Returns True on success."""

    db3_files = list(bag_folder.glob('*.db3'))
    if not db3_files:
        print(f"  ERROR: No .db3 file found in {bag_folder}")
        return False

    db3_path = db3_files[0]
    output_csv        = output_folder / f"{bag_folder.name}.csv"
    output_csv_kali   = output_folder / f"{bag_folder.name}_kalipoints.csv"

    print(f"\nProcessing: {bag_folder.name}")

    try:
        conn = sqlite3.connect(str(db3_path))
        cursor = conn.cursor()

        # Find the topic ID
        cursor.execute("SELECT id FROM topics WHERE name = ?", (TOPIC_NAME,))
        row = cursor.fetchone()
        if row is None:
            print(f"  WARNING: Topic '{TOPIC_NAME}' not found.")
            print("  Available topics:")
            for (name,) in cursor.execute("SELECT name FROM topics"):
                print(f"    - {name}")
            conn.close()
            return False

        topic_id = row[0]

        # Fetch all messages ordered by timestamp
        cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp ASC",
            (topic_id,)
        )
        rows = cursor.fetchall()
        conn.close()

        print(f"  Messages found: {len(rows)}")

        if not rows:
            print("  WARNING: No messages found for this topic.")
            return False

        ok = 0
        fail = 0

        with open(output_csv, 'w', newline='', encoding='utf-8') as f, \
             open(output_csv_kali, 'w', newline='', encoding='utf-8') as fk:

            writer = csv.writer(f)
            writer.writerow(['timestamp_ns', 'sec', 'nanosec', 'frame_id', 'x', 'y', 'z', 'is_valid'])

            fk.write('X;Y;Z\n')

            for timestamp_ns, blob in rows:
                parsed = parse_cdr_magnetometer(bytes(blob))
                if parsed is None:
                    fail += 1
                    continue

                writer.writerow([
                    timestamp_ns,
                    parsed['sec'],
                    parsed['nanosec'],
                    parsed['frame_id'],
                    parsed['x'],
                    parsed['y'],
                    parsed['z'],
                    parsed['is_valid'],
                ])

                fk.write(f"{parsed['x']:.8f};{parsed['y']:.8f};{parsed['z']:.8f}\n")

                ok += 1

        print(f"  Converted:  {ok} OK,  {fail} failed")
        print(f"  Full CSV:       {output_csv.name}")
        print(f"  Kalipoints CSV: {output_csv_kali.name}")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    script_dir = Path(__file__).parent

    bag_root   = Path(sys.argv[1]) if len(sys.argv) > 1 else script_dir / "datasets" / "0-realLifeData" / "ROS2"
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else script_dir / "datasets" / "0-realLifeData" / "CSV"

    output_dir.mkdir(parents=True, exist_ok=True)

    bag_folders = sorted(d for d in bag_root.iterdir() if d.is_dir())

    if not bag_folders:
        print(f"ERROR: No bag folders found in {bag_root}")
        return 1

    print(f"\n{'='*60}")
    print(f"ROS2 BAG → CSV Converter")
    print(f"{'='*60}")
    print(f"Input:  {bag_root}")
    print(f"Output: {output_dir}")
    print(f"Topic:  {TOPIC_NAME}")
    print(f"Found {len(bag_folders)} bag folder(s)")

    successful = 0
    for bag_folder in bag_folders:
        if bag_to_csv(bag_folder, output_dir):
            successful += 1

    print(f"\n{'='*60}")
    print(f"Summary: {successful}/{len(bag_folders)} successfully converted")
    print(f"{'='*60}\n")

    return 0 if successful == len(bag_folders) else 1


if __name__ == '__main__':
    sys.exit(main())
