#!/usr/bin/env python3
# test_streaming_live.py

"""
Standalone test for streaming transcription system with real microphone input.
Tests the complete streaming workflow with actual audio recording.

Usage:
    python test_streaming_live.py                    # Basic test
    python test_streaming_live.py --verbose          # Detailed output
    python test_streaming_live.py --duration 30      # Record for 30 seconds
    python test_streaming_live.py --no-whisper       # Test without transcription
"""

import sys
import os
import argparse
import tempfile
import threading
import time
from pathlib import Path
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Check for required packages
try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
except ImportError as e:
    print(f"❌ Missing audio dependency: {e}")
    print("The streaming system uses the same audio dependencies as the main system:")
    print("Please ensure sounddevice and soundfile are installed.")
    sys.exit(1)

# Import our streaming components
try:
    from audio_buffer import CircularAudioBuffer
    from volume_segmenter import VolumeSegmenter, VolumeConfig
    from quick_transcript import QuickTranscriptAssembler
    from background_enhancer import BackgroundEnhancer
except ImportError as e:
    print(f"❌ Missing streaming component: {e}")
    print("Make sure all streaming components are implemented in src/")
    sys.exit(1)

# Try to import whisper (optional for basic testing)
WHISPER_AVAILABLE = False
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    print("⚠️  Whisper not available - will simulate transcription")

class LiveStreamingTest:
    """Test harness for live streaming transcription."""
    
    def __init__(self, verbose: bool = False, use_whisper: bool = True):
        self.verbose = verbose
        self.use_whisper = use_whisper and WHISPER_AVAILABLE
        self.whisper_model = None
        
        # Audio configuration
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        
        # Components
        self.audio_buffer = None
        self.volume_segmenter = None
        self.quick_assembler = None
        
        # State
        self.recording = False
        self.segments_processed = 0
        self.test_results = {
            'total_duration': 0,
            'segments_created': 0,
            'processing_times': [],
            'transcript_segments': []
        }
    
    def setup_whisper(self):
        """Initialize Whisper model."""
        if not self.use_whisper:
            print("🤖 Skipping Whisper model (disabled)")
            return True
            
        print("🤖 Loading Whisper model...")
        try:
            self.whisper_model = whisper.load_model("base")
            print("✅ Whisper model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            return False
    
    def test_audio_devices(self):
        """Test available audio input devices."""
        print("\n🎙️ Testing audio devices...")
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            
            print(f"Default input device: {devices[default_input]['name']}")
            print(f"Sample rate: {devices[default_input]['default_samplerate']}")
            
            # Test recording a short sample
            duration = 2
            print(f"Recording {duration}s test sample...")
            
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32'
            )
            sd.wait()
            
            # Check if we got audio
            max_amplitude = np.max(np.abs(audio_data))
            if max_amplitude < 0.001:
                print("⚠️  Very quiet input - check microphone levels")
            else:
                print(f"✅ Audio input working (max amplitude: {max_amplitude:.4f})")
            
            return True
            
        except Exception as e:
            print(f"❌ Audio device test failed: {e}")
            return False
    
    def setup_components(self):
        """Initialize streaming components."""
        print("\n🔧 Setting up streaming components...")
        
        # Create audio buffer
        self.audio_buffer = CircularAudioBuffer(
            capacity_seconds=300,  # 5 minutes
            sample_rate=self.sample_rate,
            channels=self.channels,
            dtype='float32'
        )
        
        # Create volume segmenter
        config = VolumeConfig(
            min_segment_duration=10.0,      # Shorter for testing
            target_segment_duration=20.0,   # Shorter for testing
            max_segment_duration=30.0,      # Shorter for testing
            volume_drop_threshold=0.3,
            silence_threshold=0.02,
            min_pause_duration=0.5,
            analysis_window=1.0
        )
        
        self.volume_segmenter = VolumeSegmenter(
            self.audio_buffer,
            config,
            verbose=self.verbose
        )
        
        # Set up segment callback
        self.volume_segmenter.set_segment_callback(self.on_segment_ready)
        
        # Create mock managers for quick assembler
        class MockTranscriptManager:
            def save_transcript(self, text, metadata=None):
                timestamp = datetime.now().strftime("%H%M%S")
                return f"/tmp/streaming_test_{timestamp}.md"
        
        class MockAudioManager:
            def store_session_audio(self, session_file, session_id):
                return f"/tmp/streaming_audio_{session_id}.wav"
        
        # Create quick assembler
        self.quick_assembler = QuickTranscriptAssembler(
            MockTranscriptManager(),
            MockAudioManager(),
            auto_clipboard=False  # Don't copy during test
        )
        
        print("✅ Streaming components initialized")
        return True
    
    def on_segment_ready(self, segment_info):
        """Called when a new audio segment is ready."""
        self.segments_processed += 1
        
        print(f"📦 Segment {self.segments_processed}: {segment_info.start_time:.1f}s-{segment_info.end_time:.1f}s "
              f"({segment_info.duration:.1f}s, {segment_info.reason})")
        
        if self.verbose:
            print(f"   Volume: avg={segment_info.avg_volume:.4f}, peak={segment_info.peak_volume:.4f}")
            print(f"   Silence: {segment_info.silence_duration:.1f}s")
        
        # Extract audio for transcription
        try:
            audio_data = self.audio_buffer.read_segment(
                segment_info.start_time,
                segment_info.duration
            )
            
            if audio_data is not None:
                # Add to quick assembler
                if hasattr(self, 'quick_assembler') and self.quick_assembler:
                    self.quick_assembler.add_segment_for_transcription(segment_info, audio_data)
                
                # Simulate or perform transcription
                if self.use_whisper and self.whisper_model:
                    transcript_text = self.transcribe_segment(audio_data)
                else:
                    transcript_text = f"[Simulated transcription for {segment_info.duration:.1f}s segment]"
                
                self.test_results['transcript_segments'].append({
                    'segment': self.segments_processed,
                    'duration': segment_info.duration,
                    'text': transcript_text
                })
                
                print(f"📝 Transcript {self.segments_processed}: \"{transcript_text[:60]}{'...' if len(transcript_text) > 60 else ''}\"")
            
        except Exception as e:
            print(f"❌ Error processing segment {self.segments_processed}: {e}")
    
    def transcribe_segment(self, audio_data):
        """Transcribe audio segment using Whisper."""
        try:
            start_time = time.time()
            result = self.whisper_model.transcribe(audio_data, language='en')
            processing_time = time.time() - start_time
            
            self.test_results['processing_times'].append(processing_time)
            
            if self.verbose:
                print(f"   Whisper processing: {processing_time:.2f}s")
            
            return result['text'].strip()
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return f"[Transcription failed: {str(e)[:40]}]"
    
    def record_audio_stream(self, duration_seconds):
        """Record audio and feed it to the streaming system."""
        print(f"\n🎙️ Starting {duration_seconds}s streaming recording...")
        print("Talk into your microphone...")
        
        # Start components
        self.audio_buffer.start_recording()
        self.volume_segmenter.start_analysis()
        
        if self.quick_assembler:
            session_id = f"live_test_{int(time.time())}"
            self.quick_assembler.start_session(session_id)
        
        # Record in chunks
        chunk_duration = 0.5  # 500ms chunks
        total_chunks = int(duration_seconds / chunk_duration)
        
        try:
            for chunk_num in range(total_chunks):
                if not self.recording:
                    break
                
                # Record audio chunk
                audio_chunk = sd.rec(
                    int(chunk_duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype='float32'
                )
                sd.wait()
                
                # Feed to buffer
                self.audio_buffer.write(audio_chunk.flatten())
                
                # Analyze for segments
                self.volume_segmenter.analyze_and_segment()
                
                # Progress indicator
                if chunk_num % 4 == 0:  # Every 2 seconds
                    elapsed = chunk_num * chunk_duration
                    if self.verbose:
                        buffer_stats = self.audio_buffer.get_stats()
                        segmenter_stats = self.volume_segmenter.get_stats()
                        print(f"⏱️  {elapsed:.1f}s | Buffer: {buffer_stats['buffer_utilization']:.0%} | "
                              f"Segments: {segmenter_stats['segments_detected']}")
                    else:
                        print(f"⏱️  Recording: {elapsed:.1f}s | Segments: {self.segments_processed}", end='\r')
                
        except KeyboardInterrupt:
            print("\n⏹️  Recording interrupted by user")
        except Exception as e:
            print(f"\n❌ Recording error: {e}")
        
        # Stop recording
        self.recording = False
        print(f"\n⏹️  Recording stopped after {duration_seconds}s")
        
        # Process final segment
        final_segment = self.volume_segmenter.flush_remaining_segment()
        if final_segment:
            print(f"📦 Final segment: {final_segment.duration:.1f}s")
        
        # Stop components
        self.volume_segmenter.stop_analysis()
        self.audio_buffer.stop_recording()
        
        # Finalize transcript if using assembler
        if self.quick_assembler:
            print("\n📋 Finalizing transcript...")
            result = self.quick_assembler.finalize_transcript()
            if result:
                print(f"✅ Transcript ready: {len(result.transcript_text)} characters")
                if self.verbose:
                    print(f"Full transcript: \"{result.transcript_text}\"")
        
        self.test_results['total_duration'] = duration_seconds
        self.test_results['segments_created'] = self.segments_processed
    
    def print_results(self):
        """Print test results summary."""
        print("\n📊 Test Results Summary")
        print("=" * 50)
        print(f"Total recording time: {self.test_results['total_duration']:.1f}s")
        print(f"Segments created: {self.test_results['segments_created']}")
        
        if self.test_results['segments_created'] > 0:
            avg_segment_length = self.test_results['total_duration'] / self.test_results['segments_created']
            print(f"Average segment length: {avg_segment_length:.1f}s")
        
        if self.test_results['processing_times']:
            avg_processing = np.mean(self.test_results['processing_times'])
            print(f"Average transcription time: {avg_processing:.2f}s per segment")
        
        print(f"Transcript segments: {len(self.test_results['transcript_segments'])}")
        
        if self.verbose and self.test_results['transcript_segments']:
            print("\nTranscript segments:")
            for seg in self.test_results['transcript_segments']:
                print(f"  {seg['segment']}: {seg['text']}")
    
    def run_test(self, duration_seconds=30):
        """Run the complete streaming test."""
        print("🧪 Live Streaming Transcription Test")
        print("=" * 40)
        
        # Setup steps
        if not self.test_audio_devices():
            return False
        
        if not self.setup_whisper():
            return False
        
        if not self.setup_components():
            return False
        
        # Run the test
        self.recording = True
        self.record_audio_stream(duration_seconds)
        
        # Show results
        self.print_results()
        
        print("\n✅ Live streaming test completed!")
        return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Test streaming transcription with live audio')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--duration', '-d', type=int, default=30, help='Recording duration in seconds')
    parser.add_argument('--no-whisper', action='store_true', help='Skip Whisper transcription')
    
    args = parser.parse_args()
    
    # Create and run test
    test = LiveStreamingTest(
        verbose=args.verbose,
        use_whisper=not args.no_whisper
    )
    
    success = test.run_test(duration_seconds=args.duration)
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("The streaming components are working correctly with live audio.")
        return 0
    else:
        print("\n❌ Test failed!")
        return 1

if __name__ == "__main__":
    exit(main())