"""
VergeDB: Edge Database for IoT Analytics
Implements the architecture described in the VergeDB research paper
"""

import threading
import queue
import pickle
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import numpy as np


@dataclass
class DataSegment:
    """Represents a segment of time-series data"""
    signal_id: str
    timestamp_start: datetime
    timestamp_end: datetime
    data: np.ndarray
    segment_id: int


@dataclass
class CompressedSegment:
    """Represents a compressed segment"""
    signal_id: str
    timestamp_start: datetime
    timestamp_end: datetime
    compressed_data: bytes
    compression_method: str
    original_size: int
    compressed_size: int
    metadata: Dict[str, Any]


class AnalyticalTask(Enum):
    """Types of downstream analytical tasks"""
    AGGREGATION = "aggregation"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    SIMILARITY_SEARCH = "similarity_search"
    FORECASTING = "forecasting"


class Signal:
    """Represents a single data signal (sensor stream)"""
    
    def __init__(self, signal_id: str, segment_size: int = 1024,
                 buffer_capacity: int = 100, compression_method: str = "paa"):
        self.signal_id = signal_id
        self.segment_size = segment_size
        self.buffer_capacity = buffer_capacity
        self.compression_method = compression_method
        
        # Buffers
        self.uncompressed_buffer = queue.Queue(maxsize=buffer_capacity)
        self.compressed_buffer = queue.Queue(maxsize=buffer_capacity)
        
        # Current segment being built
        self.current_segment = []
        self.segment_counter = 0
        
        # Statistics
        self.total_ingested = 0
        self.total_compressed = 0
        self.segments_flushed_to_disk = 0
        
        # Lock for thread-safe operations
        self.lock = threading.Lock()
    
    def add_data_point(self, timestamp: datetime, value: float) -> Optional[DataSegment]:
        """Add a data point and return a segment if complete"""
        with self.lock:
            self.current_segment.append((timestamp, value))
            self.total_ingested += 1
            
            if len(self.current_segment) >= self.segment_size:
                # Create segment
                timestamps, values = zip(*self.current_segment)
                segment = DataSegment(
                    signal_id=self.signal_id,
                    timestamp_start=timestamps[0],
                    timestamp_end=timestamps[-1],
                    data=np.array(values, dtype=np.float64),
                    segment_id=self.segment_counter
                )
                self.segment_counter += 1
                self.current_segment = []
                return segment
        return None
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get buffer status"""
        return {
            'signal_id': self.signal_id,
            'uncompressed_buffer_size': self.uncompressed_buffer.qsize(),
            'compressed_buffer_size': self.compressed_buffer.qsize(),
            'current_segment_size': len(self.current_segment),
            'total_ingested': self.total_ingested,
            'total_compressed': self.total_compressed,
            'compression_method': self.compression_method
        }


class VergeDB:
    """
    VergeDB: Adaptive and task-aware compression database for IoT data
    Based on the paper "VergeDB: A Database for IoT Analytics on Edge Devices"
    """
    
    def __init__(self, data_dir: str = "data/vergedb",
                 compression_threads: int = 2,
                 segment_size: int = 1024,
                 buffer_capacity: int = 100):
        """
        Initialize VergeDB
        
        Args:
            data_dir: Directory for storing data
            compression_threads: Number of compression worker threads
            segment_size: Number of data points per segment
            buffer_capacity: Maximum segments in buffer
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.segment_size = segment_size
        self.buffer_capacity = buffer_capacity
        self.compression_threads_count = compression_threads
        
        # Multi-signal management
        self.signals: Dict[str, Signal] = {}
        
        # Compression threads
        self.compression_threads: List[threading.Thread] = []
        self.running = False
        
        # Statistics
        self.stats = {
            'total_data_points_ingested': 0,
            'total_segments_compressed': 0,
            'total_segments_flushed': 0,
            'total_original_bytes': 0,
            'total_compressed_bytes': 0,
            'start_time': time.time()
        }
        
        # Initialize compression engine
        from advanced_compression import AdvancedCompressionEngine
        self.compression_engine = AdvancedCompressionEngine()
    
    def register_signal(self, signal_id: str, compression_method: str = "paa",
                       analytical_task: Optional[AnalyticalTask] = None):
        """
        Register a new signal (data stream)
        
        Args:
            signal_id: Unique identifier for the signal
            compression_method: Compression method to use
            analytical_task: Downstream analytical task (for adaptive selection)
        """
        if signal_id in self.signals:
            print(f"⚠️  Signal {signal_id} already registered")
            return
        
        # Create signal with appropriate compression based on task
        if analytical_task:
            compression_method = self._select_compression_for_task(analytical_task)
        
        signal = Signal(
            signal_id=signal_id,
            segment_size=self.segment_size,
            buffer_capacity=self.buffer_capacity,
            compression_method=compression_method
        )
        
        self.signals[signal_id] = signal
        print(f"✅ Registered signal: {signal_id} with compression: {compression_method}")
    
    def _select_compression_for_task(self, task: AnalyticalTask) -> str:
        """Select appropriate compression method based on analytical task"""
        task_compression_map = {
            AnalyticalTask.AGGREGATION: "paa",
            AnalyticalTask.CLASSIFICATION: "block_subsample",
            AnalyticalTask.CLUSTERING: "fourier",
            AnalyticalTask.ANOMALY_DETECTION: "fourier",
            AnalyticalTask.SIMILARITY_SEARCH: "fourier",
            AnalyticalTask.FORECASTING: "block_subsample"
        }
        return task_compression_map.get(task, "paa")
    
    def ingest_data(self, signal_id: str, timestamp: datetime, value: float):
        """
        Ingest a single data point
        
        Args:
            signal_id: Signal identifier
            timestamp: Data timestamp
            value: Data value
        """
        if signal_id not in self.signals:
            raise ValueError(f"Signal {signal_id} not registered")
        
        signal = self.signals[signal_id]
        segment = signal.add_data_point(timestamp, value)
        
        if segment:
            # Try to add to uncompressed buffer
            try:
                signal.uncompressed_buffer.put_nowait(segment)
            except queue.Full:
                # Buffer full, flush to disk
                self._flush_uncompressed_buffer(signal_id)
                signal.uncompressed_buffer.put(segment)
        
        self.stats['total_data_points_ingested'] += 1
    
    def ingest_batch(self, signal_id: str, data: List[Tuple[datetime, float]]):
        """
        Ingest a batch of data points
        
        Args:
            signal_id: Signal identifier
            data: List of (timestamp, value) tuples
        """
        for timestamp, value in data:
            self.ingest_data(signal_id, timestamp, value)
    
    def start_compression(self):
        """Start compression worker threads"""
        if self.running:
            print("⚠️  Compression threads already running")
            return
        
        self.running = True
        
        for i in range(self.compression_threads_count):
            thread = threading.Thread(
                target=self._compression_worker,
                args=(i,),
                daemon=True
            )
            thread.start()
            self.compression_threads.append(thread)
        
        print(f"✅ Started {self.compression_threads_count} compression threads")
    
    def stop_compression(self):
        """Stop compression worker threads"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.compression_threads:
            thread.join(timeout=2.0)
        
        self.compression_threads.clear()
        print("✅ Stopped compression threads")
    
    def _compression_worker(self, worker_id: int):
        """Compression worker thread"""
        while self.running:
            compressed_any = False
            
            # Process each signal's uncompressed buffer
            for signal_id, signal in self.signals.items():
                try:
                    # Get segment from uncompressed buffer
                    segment = signal.uncompressed_buffer.get_nowait()
                    
                    # Compress the segment
                    compressed_segment = self._compress_segment(segment, signal.compression_method)
                    
                    # Add to compressed buffer
                    try:
                        signal.compressed_buffer.put_nowait(compressed_segment)
                        signal.total_compressed += 1
                        self.stats['total_segments_compressed'] += 1
                        self.stats['total_original_bytes'] += compressed_segment.original_size
                        self.stats['total_compressed_bytes'] += compressed_segment.compressed_size
                        compressed_any = True
                    except queue.Full:
                        # Flush compressed buffer to disk
                        self._flush_compressed_buffer(signal_id)
                        signal.compressed_buffer.put(compressed_segment)
                    
                except queue.Empty:
                    continue
            
            # Sleep briefly if no work was done
            if not compressed_any:
                time.sleep(0.01)
    
    def _compress_segment(self, segment: DataSegment, method: str) -> CompressedSegment:
        """Compress a data segment using specified method"""
        start_time = time.time()
        original_size = segment.data.nbytes
        
        compressed_data, metadata = self.compression_engine.compress(
            segment.data, method
        )
        
        compressed_size = len(compressed_data)
        metadata['compression_time'] = time.time() - start_time
        
        return CompressedSegment(
            signal_id=segment.signal_id,
            timestamp_start=segment.timestamp_start,
            timestamp_end=segment.timestamp_end,
            compressed_data=compressed_data,
            compression_method=method,
            original_size=original_size,
            compressed_size=compressed_size,
            metadata=metadata
        )
    
    def _flush_uncompressed_buffer(self, signal_id: str):
        """Flush uncompressed buffer to disk"""
        signal = self.signals[signal_id]
        flush_dir = self.data_dir / "uncompressed" / signal_id
        flush_dir.mkdir(parents=True, exist_ok=True)
        
        segments = []
        while not signal.uncompressed_buffer.empty():
            try:
                segments.append(signal.uncompressed_buffer.get_nowait())
            except queue.Empty:
                break
        
        if segments:
            filename = flush_dir / f"segment_{int(time.time()*1000)}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(segments, f)
            
            signal.segments_flushed_to_disk += len(segments)
            self.stats['total_segments_flushed'] += len(segments)
            print(f"💾 Flushed {len(segments)} uncompressed segments to disk: {signal_id}")
    
    def _flush_compressed_buffer(self, signal_id: str):
        """Flush compressed buffer to disk"""
        signal = self.signals[signal_id]
        flush_dir = self.data_dir / "compressed" / signal_id
        flush_dir.mkdir(parents=True, exist_ok=True)
        
        segments = []
        while not signal.compressed_buffer.empty():
            try:
                segments.append(signal.compressed_buffer.get_nowait())
            except queue.Empty:
                break
        
        if segments:
            filename = flush_dir / f"compressed_{int(time.time()*1000)}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(segments, f)
            
            signal.segments_flushed_to_disk += len(segments)
            self.stats['total_segments_flushed'] += len(segments)
            print(f"💾 Flushed {len(segments)} compressed segments to disk: {signal_id}")
    
    def query_raw(self, signal_id: str, start_time: datetime, end_time: datetime) -> List[Tuple]:
        """Query raw data from uncompressed buffer"""
        if signal_id not in self.signals:
            raise ValueError(f"Signal {signal_id} not registered")
        
        signal = self.signals[signal_id]
        results = []
        
        # Search in uncompressed buffer
        temp_segments = []
        while not signal.uncompressed_buffer.empty():
            segment = signal.uncompressed_buffer.get()
            temp_segments.append(segment)
            
            if segment.timestamp_start <= end_time and segment.timestamp_end >= start_time:
                # Extract data points in range
                for i, val in enumerate(segment.data):
                    # Note: This is simplified, would need actual timestamps
                    results.append((segment.timestamp_start, val))
        
        # Put segments back
        for seg in temp_segments:
            signal.uncompressed_buffer.put(seg)
        
        return results
    
    def query_compressed(self, signal_id: str, start_time: datetime, end_time: datetime) -> List[CompressedSegment]:
        """Query compressed segments"""
        if signal_id not in self.signals:
            raise ValueError(f"Signal {signal_id} not registered")
        
        signal = self.signals[signal_id]
        results = []
        
        # Search in compressed buffer
        temp_segments = []
        while not signal.compressed_buffer.empty():
            segment = signal.compressed_buffer.get()
            temp_segments.append(segment)
            
            if segment.timestamp_start <= end_time and segment.timestamp_end >= start_time:
                results.append(segment)
        
        # Put segments back
        for seg in temp_segments:
            signal.compressed_buffer.put(seg)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        runtime = time.time() - self.stats['start_time']
        
        stats = {
            'runtime_seconds': runtime,
            'total_signals': len(self.signals),
            'total_data_points_ingested': self.stats['total_data_points_ingested'],
            'total_segments_compressed': self.stats['total_segments_compressed'],
            'total_segments_flushed': self.stats['total_segments_flushed'],
            'total_original_bytes': self.stats['total_original_bytes'],
            'total_compressed_bytes': self.stats['total_compressed_bytes'],
            'ingestion_rate_per_sec': self.stats['total_data_points_ingested'] / runtime if runtime > 0 else 0,
            'signals': {}
        }
        
        for signal_id, signal in self.signals.items():
            stats['signals'][signal_id] = signal.get_buffer_status()
        
        return stats
    
    def shutdown(self):
        """Shutdown database and flush all buffers"""
        print("\n🛑 Shutting down VergeDB...")
        
        # Stop compression threads
        self.stop_compression()
        
        # Flush all buffers to disk
        for signal_id in self.signals:
            self._flush_uncompressed_buffer(signal_id)
            self._flush_compressed_buffer(signal_id)
        
        print("✅ VergeDB shutdown complete")
