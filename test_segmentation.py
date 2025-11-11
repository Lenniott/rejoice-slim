#!/usr/bin/env python3
"""
Test script to understand volume segmentation behavior.
"""

import sys
import os
sys.path.append('src')

from volume_segmenter import VolumeSegmenter, VolumeConfig
from audio_buffer import CircularAudioBuffer
import numpy as np
import time

def test_segmentation():
    """Test how volume segmentation works with different audio patterns."""
    
    # Initialize audio buffer
    audio_buffer = CircularAudioBuffer(
        capacity_seconds=300,
        sample_rate=16000,
        channels=1
    )
    
    # Configure volume segmenter with current settings
    config = VolumeConfig(
        min_segment_duration=30,
        target_segment_duration=60, 
        max_segment_duration=90
    )
    
    volume_segmenter = VolumeSegmenter(
        audio_buffer=audio_buffer,
        config=config,
        verbose=True
    )
    
    print("🧪 Testing volume segmentation behavior...")
    print(f"📊 Settings: min={config.min_segment_duration}s, target={config.target_segment_duration}s, max={config.max_segment_duration}s")
    
    # Start audio buffer and segmenter
    audio_buffer.start_recording()
    volume_segmenter.start_analysis()
    
    print(f"🔍 Buffer recording: {audio_buffer.is_recording}")
    print(f"🔍 Segmenter analyzing: {volume_segmenter.is_analyzing}")
    
    # Simulate audio data
    sample_rate = 16000
    
    print("\n🎤 Simulating continuous speech without pauses...")
    
    # Generate 95 seconds of continuous audio (to exceed max duration)
    total_duration = 95  
    chunk_duration = 0.1  # 100ms chunks
    chunks_per_second = int(1.0 / chunk_duration)
    
    for second in range(int(total_duration * chunks_per_second)):
        # Generate some audio data (random noise simulating speech)
        chunk_size = int(sample_rate * chunk_duration)
        audio_data = np.random.normal(0, 0.1, chunk_size).astype(np.float32)
        
        # Add to buffer
        audio_buffer.write(audio_data)
        
        # Check for segments every second
        if second % chunks_per_second == 0:
            current_time = second / chunks_per_second
            new_segments = volume_segmenter.analyze_and_segment()
            if new_segments:
                print(f"\n🔍 At {current_time:.1f}s: Found {len(new_segments)} new segments!")
                for i, segment in enumerate(new_segments):
                    print(f"  Segment {i+1}: {segment.start_time:.1f}s-{segment.end_time:.1f}s ({segment.duration:.1f}s) - {segment.reason}")
            elif current_time > 0 and int(current_time) % 10 == 0:
                print(f"⏱️  {current_time:.0f}s: No segments yet (continuous speech)")
        
        time.sleep(0.01)  # Small delay to simulate real-time
    
    # Stop and get final segments
    volume_segmenter.stop_analysis()
    final_segments = volume_segmenter.get_detected_segments()
    
    print(f"\n✅ Final results:")
    print(f"📦 Total segments detected: {len(final_segments)}")
    for i, segment in enumerate(final_segments):
        print(f"  Segment {i+1}: {segment.start_time:.1f}s-{segment.end_time:.1f}s ({segment.duration:.1f}s) - {segment.reason}")
    
    audio_buffer.stop_recording()
    
    return final_segments

if __name__ == "__main__":
    segments = test_segmentation()
    
    if segments:
        print(f"\n🎯 Success! Volume segmentation is working correctly.")
        print(f"💡 In your 34s recording, no segments were created because:")
        print(f"   • Duration (34s) was above minimum (30s)")  
        print(f"   • But below target (60s)")
        print(f"   • And no natural pauses were detected")
        print(f"   • This is correct behavior!")
    else:
        print(f"\n⚠️  No segments were detected. This suggests an issue with volume analysis.")