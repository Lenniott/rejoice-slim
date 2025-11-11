#!/usr/bin/env python3
"""
Debug volume segmentation max duration trigger.
"""

import sys
import os
sys.path.append('src')

from volume_segmenter import VolumeSegmenter, VolumeConfig
from audio_buffer import CircularAudioBuffer
import numpy as np
import time

def debug_max_duration():
    """Debug why max duration isn't triggering at 90s."""
    
    # Initialize with shorter durations for faster testing
    config = VolumeConfig(
        min_segment_duration=5,   # 5s for quick test
        target_segment_duration=8, # 8s 
        max_segment_duration=10   # 10s max - should DEFINITELY trigger
    )
    
    audio_buffer = CircularAudioBuffer(capacity_seconds=60, sample_rate=16000, channels=1)
    volume_segmenter = VolumeSegmenter(audio_buffer=audio_buffer, config=config, verbose=True)
    
    print(f"🧪 Testing max duration trigger with settings:")
    print(f"   Min: {config.min_segment_duration}s")
    print(f"   Target: {config.target_segment_duration}s") 
    print(f"   Max: {config.max_segment_duration}s")
    
    # Start everything
    audio_buffer.start_recording()
    volume_segmenter.start_analysis()
    
    # Generate audio for 15 seconds (should trigger max duration at 10s)
    sample_rate = 16000
    chunk_duration = 0.1  # 100ms chunks
    total_test_time = 15
    
    start_real_time = time.time()
    
    for i in range(int(total_test_time / chunk_duration)):
        # Generate audio chunk
        chunk_size = int(sample_rate * chunk_duration)
        audio_data = np.random.normal(0, 0.1, chunk_size).astype(np.float32)
        
        # Write to buffer
        audio_buffer.write(audio_data)
        
        # Get current recording duration from buffer
        buffer_duration = audio_buffer.get_recording_duration()
        
        # Run analysis every 0.5s
        if i % 5 == 0:  # Every 5 chunks = 0.5s
            new_segments = volume_segmenter.analyze_and_segment()
            elapsed = time.time() - start_real_time
            
            print(f"⏱️  {elapsed:.1f}s real | {buffer_duration:.1f}s buffer | New segments: {len(new_segments)}")
            
            if new_segments:
                for segment in new_segments:
                    print(f"   🎯 SEGMENT: {segment.start_time:.1f}s-{segment.end_time:.1f}s ({segment.duration:.1f}s) - {segment.reason}")
                    
        time.sleep(chunk_duration)
    
    # Final check
    final_segments = volume_segmenter.get_detected_segments()
    print(f"\n✅ Final result: {len(final_segments)} segments detected")
    for i, segment in enumerate(final_segments):
        print(f"   Segment {i+1}: {segment.start_time:.1f}s-{segment.end_time:.1f}s ({segment.duration:.1f}s) - {segment.reason}")
    
    volume_segmenter.stop_analysis()
    audio_buffer.stop_recording()
    
    return len(final_segments) > 0

if __name__ == "__main__":
    success = debug_max_duration()
    if success:
        print("\n🎯 Max duration trigger is working!")
    else:
        print("\n❌ Max duration trigger is NOT working - this is the bug!")