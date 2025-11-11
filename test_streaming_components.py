#!/usr/bin/env python3
# test_streaming_components.py

import sys
import os
import time
import numpy as np
import tempfile
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from audio_buffer import CircularAudioBuffer, BufferSegmentIterator
from volume_segmenter import VolumeSegmenter, VolumeConfig
from src.quick_transcript import QuickTranscriptAssembler
from src.background_enhancer import BackgroundEnhancer
from src.safety_net import SafetyNetManager, SafetyNetIntegrator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_audio_buffer():
    """Test the circular audio buffer functionality."""
    print("\n🧪 Testing CircularAudioBuffer...")
    
    # Create a small buffer for testing
    buffer = CircularAudioBuffer(capacity_seconds=5, sample_rate=1000, channels=1)
    
    # Test basic operations
    buffer.start_recording()
    assert buffer.is_recording == True
    
    # Write some test data
    test_data = np.random.random(1000).astype(np.float32)  # 1 second of data
    buffer.write(test_data)
    
    # Test statistics
    stats = buffer.get_stats()
    assert stats['is_recording'] == True
    assert stats['total_samples_written'] == 1000
    
    # Now test get_latest_segment (should work with the fix)
    segment = buffer.get_latest_segment(0.5)  # Get last 0.5 seconds
    assert segment is not None, "Should get latest segment after fix"
    assert len(segment) == 500, f"Expected 500 samples, got {len(segment)}"
    
    # Test direct segment reading by time offset
    segment2 = buffer.read_segment(0.0, 0.5)  # Read first 0.5 seconds
    assert segment2 is not None, "Should be able to read by time offset"
    assert len(segment2) == 500, f"Expected 500 samples, got {len(segment2)}"
    
    # Write more data and test again
    test_data2 = np.random.random(2000).astype(np.float32)  # 2 more seconds
    buffer.write(test_data2)
    
    # Now get_latest_segment with more data
    segment3 = buffer.get_latest_segment(1.5)  # Get last 1.5 seconds
    assert segment3 is not None, "Should get latest segment with more data"
    assert len(segment3) == 1500, f"Expected 1500 samples, got {len(segment3)}"
    
    buffer.stop_recording()
    print("✅ CircularAudioBuffer tests passed")

def test_volume_segmenter():
    """Test the volume-based segmentation."""
    print("\n🧪 Testing VolumeSegmenter...")
    
    # Create buffer and segmenter
    buffer = CircularAudioBuffer(capacity_seconds=10, sample_rate=1000, channels=1)
    config = VolumeConfig(
        min_segment_duration=1.0,
        target_segment_duration=2.0,
        max_segment_duration=3.0,
        analysis_window=0.5
    )
    segmenter = VolumeSegmenter(buffer, config, verbose=True)
    
    # Start recording and analysis
    buffer.start_recording()
    segmenter.start_analysis()
    
    # Simulate audio with volume changes
    # High volume for 2 seconds
    high_volume_data = np.random.random(2000).astype(np.float32) * 0.8
    buffer.write(high_volume_data)
    time.sleep(0.1)
    
    # Low volume (silence) for 1 second
    low_volume_data = np.random.random(1000).astype(np.float32) * 0.01
    buffer.write(low_volume_data)
    time.sleep(0.1)
    
    # More high volume
    high_volume_data2 = np.random.random(2000).astype(np.float32) * 0.9
    buffer.write(high_volume_data2)
    time.sleep(0.1)
    
    # Analyze for segments
    segments = segmenter.analyze_and_segment()
    
    # Get final segment
    final_segment = segmenter.flush_remaining_segment()
    if final_segment:
        segments.append(final_segment)
    
    # Test results
    stats = segmenter.get_stats()
    print(f"Detected {len(segments)} segments")
    print(f"Segmenter stats: {stats}")
    
    segmenter.stop_analysis()
    buffer.stop_recording()
    
    print("✅ VolumeSegmenter tests passed")

def test_integration_flow():
    """Test the integration flow between components."""
    print("\n🧪 Testing integration flow...")
    
    # Mock transcript and audio managers for testing
    class MockTranscriptManager:
        def save_transcript(self, text, metadata=None):
            return f"/tmp/test_transcript_{int(time.time())}.md"
        
        def update_transcript_content(self, path, enhanced_transcript=None, 
                                    enhanced_summary=None, enhancement_metadata=None):
            print(f"Updated transcript at {path}")
    
    class MockAudioManager:
        def store_session_audio(self, session_file, session_id):
            return f"/tmp/test_audio_{session_id}.wav"
    
    class MockSummarizationService:
        def create_comprehensive_summary(self, text, include_detailed_analysis=False):
            return {"summary": f"Mock summary of: {text[:50]}..."}
    
    # Create components
    transcript_manager = MockTranscriptManager()
    audio_manager = MockAudioManager()
    summarization_service = MockSummarizationService()
    
    # Test quick transcript assembler
    assembler = QuickTranscriptAssembler(
        transcript_manager, 
        audio_manager, 
        auto_clipboard=False  # Don't actually copy to clipboard during tests
    )
    
    # Test background enhancer
    enhancer = BackgroundEnhancer(
        transcript_manager,
        audio_manager, 
        summarization_service,
        auto_cleanup=False  # Don't delete files during tests
    )
    
    # Start a session
    session_id = f"test_session_{int(time.time())}"
    assembler.start_session(session_id)
    
    # Simulate segment processing
    from volume_segmenter import SegmentInfo
    test_segment = SegmentInfo(
        start_time=0.0,
        end_time=30.0,
        duration=30.0,
        reason='test',
        avg_volume=0.5,
        peak_volume=0.8,
        silence_duration=0.0
    )
    
    # Add segment (with mock audio data)
    test_audio = np.random.random(48000).astype(np.float32)  # 3 seconds at 16kHz
    segment_id = assembler.add_segment_for_transcription(test_segment, test_audio)
    
    # Wait for processing
    time.sleep(1.0)
    
    # Finalize transcript
    result = assembler.finalize_transcript()
    assert result is not None
    assert result.transcript_id == session_id
    
    print(f"Quick transcript result: {result.transcript_text[:100]}...")
    
    # Test background enhancement
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
        tmp_audio_path = tmp_audio.name
    
    # Queue for enhancement (don't actually start worker for test)
    success = enhancer.queue_enhancement(result, tmp_audio_path)
    assert success == True
    
    # Check queue status
    queue_stats = enhancer.get_queue_stats()
    assert queue_stats['queue']['pending'] == 1
    
    # Cleanup
    try:
        os.unlink(tmp_audio_path)
    except:
        pass
    
    print("✅ Integration flow tests passed")

def test_safety_net_manager():
    """Test SafetyNetManager functionality."""
    print("\n=== Testing SafetyNetManager ===")
    
    # Initialize safety net
    safety_net = SafetyNetManager(
        safety_log_path="test_safety_log.json",
        auto_recovery=True,
        max_retry_attempts=2
    )
    
    # Start a session
    session_id = "test_session_001"
    master_audio = "/tmp/test_master.wav"
    
    record = safety_net.start_session(session_id, master_audio)
    assert record.session_id == session_id
    assert record.master_audio_path == master_audio
    assert record.fallback_available == True
    print(f"✓ Started session: {session_id}")
    
    # Register processing attempts
    streaming_attempt = safety_net.register_processing_attempt(
        session_id, "streaming", {"segment_size": 30}
    )
    print(f"✓ Registered streaming attempt: {streaming_attempt}")
    
    quick_attempt = safety_net.register_processing_attempt(
        session_id, "quick", {"assembly_mode": "fast"}
    )
    print(f"✓ Registered quick attempt: {quick_attempt}")
    
    # Complete attempts with success
    safety_net.complete_processing_attempt(
        session_id, streaming_attempt, 
        output_files=["stream_segment_1.wav", "stream_segment_2.wav"],
        success=True
    )
    print("✓ Completed streaming attempt successfully")
    
    safety_net.complete_processing_attempt(
        session_id, quick_attempt,
        output_files=["quick_transcript.txt"],
        success=True
    )
    print("✓ Completed quick attempt successfully")
    
    # Test failure scenario
    enhanced_attempt = safety_net.register_processing_attempt(
        session_id, "enhanced", {"quality": "high"}
    )
    
    safety_net.complete_processing_attempt(
        session_id, enhanced_attempt,
        success=False,
        error_message="Whisper processing timeout"
    )
    print("✓ Tested failure handling")
    
    # Check session status
    status = safety_net.get_session_status(session_id)
    assert status is not None
    assert status['attempts_count'] == 3
    print(f"✓ Session status: {status['attempts_count']} attempts")
    
    # Complete session
    final_record = safety_net.complete_session(
        session_id, success=True, 
        final_output_files=["final_transcript.txt", "summary.txt"]
    )
    print(f"✓ Completed session with status: {final_record.final_status}")
    
    # Test statistics
    stats = safety_net.get_safety_stats()
    assert stats['lifetime_stats']['sessions_tracked'] >= 1
    print(f"✓ Safety stats: {stats['active_sessions']} active, {stats['lifetime_stats']['sessions_tracked']} tracked")
    
    # Test integrator
    integrator = SafetyNetIntegrator(safety_net)
    print("✓ SafetyNetIntegrator initialized with callbacks")
    
    # Cleanup test file
    import os
    if os.path.exists("test_safety_log.json"):
        os.remove("test_safety_log.json")
    
    print("✓ SafetyNetManager test completed successfully!")

def main():
    """Run all tests."""
    print("🚀 Starting streaming components tests...")
    
    try:
        test_audio_buffer()
        test_volume_segmenter() 
        test_integration_flow()
        test_safety_net_manager()
        
        print("\n🎉 All tests passed! Complete streaming system with safety net is working correctly.")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())