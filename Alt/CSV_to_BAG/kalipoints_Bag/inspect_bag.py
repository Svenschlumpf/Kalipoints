#!/usr/bin/env python3
import rosbag
import sys

if len(sys.argv) < 2:
    print("Usage: python3 inspect_bag.py <bag_file>")
    sys.exit(1)

bag_file = sys.argv[1]

with rosbag.Bag(bag_file) as bag:
    print(f"\n== BAG File: {bag_file} ==\n")
    
    # Get topics
    topics = bag.get_type_and_topic_info()[1]
    print(f"Topics found:")
    for topic, info in topics.items():
        print(f"  - Topic: {topic}")
        print(f"    Message type: {info.msg_type}")
        print(f"    Message count: {info.message_count}")
    
    print(f"\n== First 5 messages ==\n")
    
    msg_count = 0
    for topic, msg, t in bag.read_messages():
        if msg_count >= 5:
            break
        
        print(f"Topic: {topic}")
        print(f"Timestamp: {t}")
        print(f"Message: {msg}")
        print("---")
        msg_count += 1

