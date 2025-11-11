# src/safety_net.py

import threading
import time
import json
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProcessingAttempt:
    """Record of a processing attempt."""
    attempt_id: str
    stage: str  # 'streaming', 'quick', 'enhanced'
    start_time: float
    end_time: Optional[float]
    status: str  # 'pending', 'processing', 'completed', 'failed'
    error_message: Optional[str]
    output_files: List[str]
    metadata: Dict[str, Any]

@dataclass
class SafetyRecord:
    """Complete safety record for a session."""
    session_id: str
    created_at: datetime
    master_audio_path: Optional[str]
    processing_attempts: List[ProcessingAttempt]
    current_stage: str
    fallback_available: bool
    recovery_performed: bool
    final_status: str  # 'success', 'partial', 'failed'

class SafetyNetManager:
    """
    Triple redundancy manager for data preservation and recovery.
    
    Tracks all processing stages, provides fallback chains,
    and ensures no data is ever lost.
    """
    
    def __init__(self, 
                 safety_log_path: str = "safety_log.json",
                 auto_recovery: bool = True,
                 max_retry_attempts: int = 3):
        """
        Initialize the safety net manager.
        
        Args:
            safety_log_path: Path to safety log file
            auto_recovery: Automatically attempt recovery on failures
            max_retry_attempts: Maximum retry attempts per stage
        """
        self.safety_log_path = Path(safety_log_path)
        self.auto_recovery = auto_recovery
        self.max_retry_attempts = max_retry_attempts
        
        # State management
        self.lock = threading.RLock()
        self.active_sessions: Dict[str, SafetyRecord] = {}
        self.recovery_callbacks: Dict[str, Callable] = {}
        
        # Statistics
        self.stats = {
            'sessions_tracked': 0,
            'recoveries_performed': 0,
            'failures_prevented': 0,
            'data_preserved': 0
        }
        
        # Load existing safety records
        self._load_safety_log()
        
        logger.info(f"SafetyNetManager initialized: auto_recovery={auto_recovery}, "
                   f"max_retries={max_retry_attempts}")
    
    def start_session(self, 
                     session_id: str, 
                     master_audio_path: Optional[str] = None) -> SafetyRecord:
        """
        Start tracking a new session.
        
        Args:
            session_id: Unique session identifier
            master_audio_path: Path to master audio file (if available)
            
        Returns:
            SafetyRecord for the session
        """
        with self.lock:
            safety_record = SafetyRecord(
                session_id=session_id,
                created_at=datetime.now(),
                master_audio_path=master_audio_path,
                processing_attempts=[],
                current_stage='streaming',
                fallback_available=master_audio_path is not None,
                recovery_performed=False,
                final_status='pending'
            )
            
            self.active_sessions[session_id] = safety_record
            self.stats['sessions_tracked'] += 1
            
        self._save_safety_log()
        logger.info(f"SafetyNet: Started tracking session {session_id}")
        return safety_record
    
    def register_processing_attempt(self, 
                                  session_id: str, 
                                  stage: str,
                                  metadata: Optional[Dict] = None) -> str:
        """
        Register a new processing attempt.
        
        Args:
            session_id: Session identifier
            stage: Processing stage ('streaming', 'quick', 'enhanced')
            metadata: Additional metadata for the attempt
            
        Returns:
            Attempt ID for tracking
        """
        attempt_id = f"{session_id}_{stage}_{int(time.time())}"
        
        with self.lock:
            if session_id not in self.active_sessions:
                logger.warning(f"Session {session_id} not tracked - creating safety record")
                self.start_session(session_id)
            
            attempt = ProcessingAttempt(
                attempt_id=attempt_id,
                stage=stage,
                start_time=time.time(),
                end_time=None,
                status='processing',
                error_message=None,
                output_files=[],
                metadata=metadata or {}
            )
            
            self.active_sessions[session_id].processing_attempts.append(attempt)
            self.active_sessions[session_id].current_stage = stage
        
        self._save_safety_log()
        logger.debug(f"SafetyNet: Registered {stage} attempt {attempt_id}")
        return attempt_id
    
    def complete_processing_attempt(self, 
                                  session_id: str, 
                                  attempt_id: str,
                                  output_files: Optional[List[str]] = None,
                                  success: bool = True,
                                  error_message: Optional[str] = None) -> None:
        """
        Mark a processing attempt as completed.
        
        Args:
            session_id: Session identifier
            attempt_id: Attempt identifier
            output_files: List of output file paths
            success: Whether the attempt succeeded
            error_message: Error message if failed
        """
        with self.lock:
            if session_id not in self.active_sessions:
                logger.warning(f"Session {session_id} not found for attempt completion")
                return
            
            # Find and update the attempt
            for attempt in self.active_sessions[session_id].processing_attempts:
                if attempt.attempt_id == attempt_id:
                    attempt.end_time = time.time()
                    attempt.status = 'completed' if success else 'failed'
                    attempt.output_files = output_files or []
                    attempt.error_message = error_message
                    break
            else:
                logger.warning(f"Attempt {attempt_id} not found for completion")
                return
            
            # Check if recovery is needed
            if not success and self.auto_recovery:
                self._trigger_recovery(session_id, attempt.stage, error_message)
        
        self._save_safety_log()
        status = 'completed' if success else 'failed'
        logger.debug(f"SafetyNet: Marked attempt {attempt_id} as {status}")
    
    def _trigger_recovery(self, 
                         session_id: str, 
                         failed_stage: str, 
                         error_message: Optional[str]) -> None:
        """
        Trigger recovery procedures for a failed attempt.
        
        Args:
            session_id: Session identifier
            failed_stage: Stage that failed
            error_message: Error details
        """
        logger.warning(f"SafetyNet: Triggering recovery for {session_id}, failed stage: {failed_stage}")
        
        safety_record = self.active_sessions[session_id]
        
        # Count previous attempts for this stage
        stage_attempts = [a for a in safety_record.processing_attempts 
                         if a.stage == failed_stage]
        
        if len(stage_attempts) >= self.max_retry_attempts:
            logger.error(f"SafetyNet: Maximum retry attempts exceeded for {failed_stage}")
            self._escalate_to_fallback(session_id, failed_stage)
            return
        
        # Attempt recovery based on stage
        recovery_plan = self._get_recovery_plan(failed_stage, error_message)
        
        if recovery_plan and 'callback' in recovery_plan:
            callback = self.recovery_callbacks.get(recovery_plan['callback'])
            if callback:
                try:
                    logger.info(f"SafetyNet: Executing recovery: {recovery_plan['description']}")
                    callback(session_id, failed_stage, recovery_plan)
                    safety_record.recovery_performed = True
                    self.stats['recoveries_performed'] += 1
                except Exception as e:
                    logger.error(f"SafetyNet: Recovery callback failed: {e}")
                    self._escalate_to_fallback(session_id, failed_stage)
    
    def _get_recovery_plan(self, failed_stage: str, error_message: Optional[str]) -> Optional[Dict]:
        """
        Generate a recovery plan based on the failure.
        
        Args:
            failed_stage: Stage that failed
            error_message: Error details
            
        Returns:
            Recovery plan dictionary or None
        """
        if failed_stage == 'streaming':
            return {
                'callback': 'retry_streaming',
                'description': 'Retry streaming with reduced segment size',
                'params': {'reduce_segment_size': True}
            }
        elif failed_stage == 'quick':
            return {
                'callback': 'fallback_to_simple_assembly',
                'description': 'Use simple segment assembly without advanced processing',
                'params': {'simple_mode': True}
            }
        elif failed_stage == 'enhanced':
            return {
                'callback': 'retry_with_smaller_chunks',
                'description': 'Retry enhancement with smaller audio chunks',
                'params': {'chunk_size_reduction': 0.5}
            }
        
        return None
    
    def _escalate_to_fallback(self, session_id: str, failed_stage: str) -> None:
        """
        Escalate to fallback procedures when recovery fails.
        
        Args:
            session_id: Session identifier
            failed_stage: Stage that failed
        """
        logger.warning(f"SafetyNet: Escalating to fallback for {session_id}, stage: {failed_stage}")
        
        safety_record = self.active_sessions[session_id]
        
        if safety_record.fallback_available and safety_record.master_audio_path:
            # Use master audio for legacy processing
            fallback_plan = {
                'callback': 'fallback_to_legacy',
                'description': f'Fallback to legacy processing for {failed_stage}',
                'params': {
                    'master_audio_path': safety_record.master_audio_path,
                    'failed_stage': failed_stage
                }
            }
            
            callback = self.recovery_callbacks.get('fallback_to_legacy')
            if callback:
                try:
                    logger.info(f"SafetyNet: Executing fallback: {fallback_plan['description']}")
                    callback(session_id, failed_stage, fallback_plan)
                    self.stats['failures_prevented'] += 1
                except Exception as e:
                    logger.error(f"SafetyNet: Fallback failed: {e}")
                    safety_record.final_status = 'failed'
            else:
                logger.error("SafetyNet: No fallback callback registered")
                safety_record.final_status = 'failed'
        else:
            logger.error(f"SafetyNet: No fallback available for {session_id}")
            safety_record.final_status = 'failed'
    
    def complete_session(self, 
                        session_id: str, 
                        success: bool = True,
                        final_output_files: Optional[List[str]] = None) -> SafetyRecord:
        """
        Mark a session as completed.
        
        Args:
            session_id: Session identifier
            success: Whether the session completed successfully
            final_output_files: Final output files produced
            
        Returns:
            Final SafetyRecord
        """
        with self.lock:
            if session_id not in self.active_sessions:
                logger.warning(f"Session {session_id} not found for completion")
                return None
            
            safety_record = self.active_sessions[session_id]
            safety_record.final_status = 'success' if success else 'failed'
            
            # Add final output files to the last successful attempt
            if final_output_files:
                for attempt in reversed(safety_record.processing_attempts):
                    if attempt.status == 'completed':
                        attempt.output_files.extend(final_output_files)
                        break
            
            # Archive the session
            del self.active_sessions[session_id]
            
        self._save_safety_log()
        logger.info(f"SafetyNet: Completed session {session_id} with status: {safety_record.final_status}")
        return safety_record
    
    def register_recovery_callback(self, callback_name: str, callback: Callable) -> None:
        """
        Register a recovery callback function.
        
        Args:
            callback_name: Name of the callback
            callback: Callback function
        """
        self.recovery_callbacks[callback_name] = callback
        logger.info(f"SafetyNet: Registered recovery callback: {callback_name}")
    
    def get_session_status(self, session_id: str) -> Optional[Dict]:
        """
        Get the current status of a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Status dictionary or None
        """
        with self.lock:
            if session_id not in self.active_sessions:
                return None
            
            safety_record = self.active_sessions[session_id]
            
            return {
                'session_id': session_id,
                'current_stage': safety_record.current_stage,
                'attempts_count': len(safety_record.processing_attempts),
                'fallback_available': safety_record.fallback_available,
                'recovery_performed': safety_record.recovery_performed,
                'status': safety_record.final_status,
                'last_attempt': safety_record.processing_attempts[-1] if safety_record.processing_attempts else None
            }
    
    def _save_safety_log(self) -> None:
        """Save safety records to persistent storage."""
        try:
            # Prepare data for serialization
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats,
                'active_sessions': {}
            }
            
            for session_id, record in self.active_sessions.items():
                log_data['active_sessions'][session_id] = {
                    'session_id': record.session_id,
                    'created_at': record.created_at.isoformat(),
                    'master_audio_path': record.master_audio_path,
                    'current_stage': record.current_stage,
                    'fallback_available': record.fallback_available,
                    'recovery_performed': record.recovery_performed,
                    'final_status': record.final_status,
                    'processing_attempts': [asdict(attempt) for attempt in record.processing_attempts]
                }
            
            # Write to file
            with open(self.safety_log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"SafetyNet: Failed to save safety log: {e}")
    
    def _load_safety_log(self) -> None:
        """Load safety records from persistent storage."""
        if not self.safety_log_path.exists():
            logger.info("SafetyNet: No existing safety log found")
            return
        
        try:
            with open(self.safety_log_path, 'r') as f:
                log_data = json.load(f)
            
            # Restore statistics
            if 'stats' in log_data:
                self.stats.update(log_data['stats'])
            
            # Restore active sessions
            if 'active_sessions' in log_data:
                for session_id, session_data in log_data['active_sessions'].items():
                    record = SafetyRecord(
                        session_id=session_data['session_id'],
                        created_at=datetime.fromisoformat(session_data['created_at']),
                        master_audio_path=session_data.get('master_audio_path'),
                        processing_attempts=[],
                        current_stage=session_data['current_stage'],
                        fallback_available=session_data['fallback_available'],
                        recovery_performed=session_data['recovery_performed'],
                        final_status=session_data['final_status']
                    )
                    
                    # Restore attempts
                    for attempt_data in session_data.get('processing_attempts', []):
                        attempt = ProcessingAttempt(**attempt_data)
                        record.processing_attempts.append(attempt)
                    
                    self.active_sessions[session_id] = record
            
            logger.info(f"SafetyNet: Loaded {len(self.active_sessions)} active sessions from safety log")
            
        except Exception as e:
            logger.error(f"SafetyNet: Failed to load safety log: {e}")
    
    def get_safety_stats(self) -> Dict:
        """Get safety net statistics."""
        with self.lock:
            return {
                'active_sessions': len(self.active_sessions),
                'lifetime_stats': self.stats.copy(),
                'recovery_callbacks': list(self.recovery_callbacks.keys()),
                'auto_recovery_enabled': self.auto_recovery,
                'max_retry_attempts': self.max_retry_attempts
            }
    
    def cleanup_old_sessions(self, hours: int = 24) -> int:
        """
        Clean up completed sessions older than specified hours.
        
        Args:
            hours: Hours threshold for cleanup
            
        Returns:
            Number of sessions cleaned up
        """
        # This would typically clean up archived sessions
        # For now, just report that no cleanup was needed
        logger.info(f"SafetyNet: Cleanup check completed (threshold: {hours}h)")
        return 0

class SafetyNetIntegrator:
    """
    Helper class for integrating SafetyNet with streaming components.
    """
    
    def __init__(self, safety_net: SafetyNetManager):
        """Initialize with SafetyNetManager instance."""
        self.safety_net = safety_net
        self._setup_recovery_callbacks()
    
    def _setup_recovery_callbacks(self):
        """Setup standard recovery callbacks."""
        self.safety_net.register_recovery_callback('retry_streaming', self._retry_streaming)
        self.safety_net.register_recovery_callback('fallback_to_simple_assembly', self._simple_assembly)
        self.safety_net.register_recovery_callback('retry_with_smaller_chunks', self._retry_smaller_chunks)
        self.safety_net.register_recovery_callback('fallback_to_legacy', self._fallback_to_legacy)
    
    def _retry_streaming(self, session_id: str, stage: str, plan: Dict):
        """Recovery callback for streaming failures."""
        logger.info(f"SafetyNet: Retrying streaming for {session_id}")
        # TODO: Implement streaming retry logic
    
    def _simple_assembly(self, session_id: str, stage: str, plan: Dict):
        """Recovery callback for assembly failures."""
        logger.info(f"SafetyNet: Using simple assembly for {session_id}")
        # TODO: Implement simple assembly logic
    
    def _retry_smaller_chunks(self, session_id: str, stage: str, plan: Dict):
        """Recovery callback for enhancement failures."""
        logger.info(f"SafetyNet: Retrying with smaller chunks for {session_id}")
        # TODO: Implement chunk size reduction logic
    
    def _fallback_to_legacy(self, session_id: str, stage: str, plan: Dict):
        """Recovery callback for complete fallback."""
        logger.info(f"SafetyNet: Falling back to legacy processing for {session_id}")
        # TODO: Implement legacy processing fallback