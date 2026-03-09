import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from rlvds.ingestion.video_source import VideoSource
from rlvds.ingestion.frame_buffer import FrameBuffer

def main():
    print("Testing VideoSource initialization and logging ...")
    
    # 1. Test VideoSource with invalid source to trigger error logging
    print("\n--- Test 1: Invalid Source ---")
    try:
        with VideoSource("invalid_video.xyz", max_read_failures=2, reconnect_interval_sec=0.1) as src:
            pass
    except Exception as e:
        print(f"Exception caught (expected): {e}")

    # 2. Test VideoSource with a real dummy video file using cv2.VideoWriter
    print("\n--- Test 2: Valid Source & Frame Buffer ---")
    dummy_video_path = "test_dummy.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(dummy_video_path, fourcc, 10.0, (640, 480))
    for i in range(15):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    out.release()

    fb = FrameBuffer(max_size=5)
    try:
        with VideoSource(dummy_video_path) as src:
            print(f"FPS: {src.get_fps()}, Size: {src.get_frame_size()}, Opened: {src.is_opened()}")
            
            generator = fb.skip_frames(src, skip=3) # Process every 3rd frame (0, 3, 6, 9, 12) -> 5 frames
            processed_count = 0
            for frame in generator:
                fb.put(frame)
                processed_count += 1
            
            print(f"Frames processed (after skipping): {processed_count}")
            print(f"Buffer full? {fb.is_full()}")
            
            latest = fb.get()
            if latest is not None:
                print(f"Latest frame shape: {latest.shape}")
                
    finally:
        if os.path.exists(dummy_video_path):
            os.remove(dummy_video_path)
            
    # Read the log file if it was generated
    log_path = "logs/rlvds.log"
    print("\n--- Log File Contents ---")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            print(f.read())
    else:
        print("No log file found.")

if __name__ == "__main__":
    main()
