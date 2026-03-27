"""
Query Engine for VergeDB
Supports analytics directly on compressed data
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import pickle


class QueryType(Enum):
    """Types of supported queries"""
    RANGE_FILTER = "range_filter"
    AGGREGATION = "aggregation"
    SIMILARITY_SEARCH = "similarity_search"
    ANOMALY_DETECTION = "anomaly_detection"


class QueryEngine:
    """
    Query engine that can operate directly on compressed data
    Supports query-friendly compression formats
    """
    
    def __init__(self):
        """Initialize query engine"""
        # Methods that support direct querying
        self.query_friendly_methods = {
            'paa': True,
            'fourier': True,
            'splitdouble': True,
            'block_subsample': True,
            'uniform_subsample': True,
            'gorilla': False,  # Requires partial decompression
            'sprintz': False,  # Requires partial decompression
            'gzip': False,  # Requires full decompression
            'snappy': False  # Requires full decompression
        }
    
    def can_query_compressed(self, compression_method: str) -> bool:
        """Check if method supports direct querying"""
        return self.query_friendly_methods.get(compression_method, False)
    
    def range_filter(self, compressed_data: bytes, compression_method: str,
                    min_val: float, max_val: float) -> Optional[np.ndarray]:
        """
        Apply range filter on compressed data
        
        Args:
            compressed_data: Compressed segment data
            compression_method: Compression method used
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Filtered data or None if full decompression needed
        """
        if compression_method == 'paa':
            return self._range_filter_paa(compressed_data, min_val, max_val)
        
        elif compression_method == 'splitdouble':
            return self._range_filter_splitdouble(compressed_data, min_val, max_val)
        
        elif compression_method == 'fourier':
            # For Fourier, we need to check magnitude
            return self._range_filter_fourier(compressed_data, min_val, max_val)
        
        else:
            # Requires full decompression
            return None
    
    def _range_filter_paa(self, compressed_data: bytes, min_val: float, max_val: float) -> np.ndarray:
        """Range filter on PAA compressed data"""
        paa_array, metadata = pickle.loads(compressed_data)
        
        # Filter PAA segments
        mask = (paa_array >= min_val) & (paa_array <= max_val)
        filtered = paa_array[mask]
        
        # Reconstruct filtered segments
        n = metadata['original_length']
        segments = metadata['segments']
        segment_size = metadata['segment_size']
        
        result = []
        for i, (val, keep) in enumerate(zip(paa_array, mask)):
            if keep:
                start = int(i * segment_size)
                end = int((i + 1) * segment_size)
                result.extend([val] * (end - start))
        
        return np.array(result)
    
    def _range_filter_splitdouble(self, compressed_data: bytes, min_val: float, max_val: float) -> np.ndarray:
        """Range filter on SplitDouble compressed data"""
        compressed = pickle.loads(compressed_data)
        
        # Can filter on integer representation
        precision = compressed['precision']
        multiplier = 10 ** precision
        
        min_int = int(min_val * multiplier)
        max_int = int(max_val * multiplier)
        
        # Reconstruct and filter
        deltas = compressed['deltas']
        first = compressed['first']
        integers = np.cumsum(deltas)
        
        mask = (integers >= min_int) & (integers <= max_int)
        filtered = integers[mask].astype(np.float64) / multiplier
        
        return filtered
    
    def _range_filter_fourier(self, compressed_data: bytes, min_val: float, max_val: float) -> np.ndarray:
        """Range filter on Fourier compressed data (approximate)"""
        # For Fourier, this is an approximation
        # In practice, we'd need domain knowledge
        compressed = pickle.loads(compressed_data)
        
        # Reconstruct
        n = compressed['length']
        fft_coeffs = np.zeros(n, dtype=np.complex128)
        fft_coeffs[compressed['indices']] = compressed['coeffs']
        
        reconstructed = np.fft.ifft(fft_coeffs)
        real_values = np.real(reconstructed)
        
        # Filter
        mask = (real_values >= min_val) & (real_values <= max_val)
        return real_values[mask]
    
    def aggregate(self, compressed_data: bytes, compression_method: str,
                 agg_func: str = 'mean') -> Optional[float]:
        """
        Perform aggregation on compressed data without full decompression
        
        Args:
            compressed_data: Compressed segment
            compression_method: Compression method
            agg_func: Aggregation function (mean, sum, min, max, std)
            
        Returns:
            Aggregated value or None if not supported
        """
        if compression_method == 'paa':
            return self._aggregate_paa(compressed_data, agg_func)
        
        elif compression_method == 'fourier':
            return self._aggregate_fourier(compressed_data, agg_func)
        
        else:
            return None
    
    def _aggregate_paa(self, compressed_data: bytes, agg_func: str) -> float:
        """Aggregate on PAA data"""
        paa_array, metadata = pickle.loads(compressed_data)
        
        if agg_func == 'mean':
            return float(np.mean(paa_array))
        elif agg_func == 'sum':
            # Approximate sum by scaling
            segment_size = metadata['segment_size']
            return float(np.sum(paa_array) * segment_size / metadata['segments'])
        elif agg_func == 'min':
            return float(np.min(paa_array))
        elif agg_func == 'max':
            return float(np.max(paa_array))
        elif agg_func == 'std':
            return float(np.std(paa_array))
        else:
            raise ValueError(f"Unknown aggregation function: {agg_func}")
    
    def _aggregate_fourier(self, compressed_data: bytes, agg_func: str) -> float:
        """Aggregate on Fourier data"""
        compressed = pickle.loads(compressed_data)
        
        if agg_func == 'mean':
            # DC component (frequency 0) represents mean
            if 0 in compressed['indices']:
                idx = np.where(compressed['indices'] == 0)[0][0]
                return float(np.real(compressed['coeffs'][idx])) / compressed['length']
            return 0.0
        
        elif agg_func == 'energy':
            # Total energy from Parseval's theorem
            return float(np.sum(np.abs(compressed['coeffs'])**2))
        
        else:
            # For other aggregations, approximate reconstruction needed
            return None
    
    def similarity_search(self, query_compressed: bytes, candidate_compressed: bytes,
                         compression_method: str, distance_metric: str = 'euclidean') -> float:
        """
        Compute similarity between compressed time series
        
        Args:
            query_compressed: Query time series (compressed)
            candidate_compressed: Candidate time series (compressed)
            compression_method: Compression method
            distance_metric: Distance metric (euclidean, dtw, cosine)
            
        Returns:
            Distance value
        """
        if compression_method in ['paa', 'fourier']:
            return self._similarity_compressed(
                query_compressed, candidate_compressed, 
                compression_method, distance_metric
            )
        else:
            return float('inf')
    
    def _similarity_compressed(self, query_data: bytes, candidate_data: bytes,
                              method: str, metric: str) -> float:
        """Compute similarity on compressed representations"""
        if method == 'paa':
            query_paa, _ = pickle.loads(query_data)
            candidate_paa, _ = pickle.loads(candidate_data)
            
            if metric == 'euclidean':
                return float(np.linalg.norm(query_paa - candidate_paa))
            elif metric == 'cosine':
                dot = np.dot(query_paa, candidate_paa)
                norm = np.linalg.norm(query_paa) * np.linalg.norm(candidate_paa)
                return 1.0 - (dot / norm) if norm > 0 else 1.0
        
        elif method == 'fourier':
            query_comp = pickle.loads(query_data)
            cand_comp = pickle.loads(candidate_data)
            
            # Compare frequency components
            q_coeffs = query_comp['coeffs']
            c_coeffs = cand_comp['coeffs']
            
            if metric == 'euclidean':
                return float(np.linalg.norm(q_coeffs - c_coeffs))
        
        return float('inf')
    
    def detect_anomalies(self, compressed_data: bytes, compression_method: str,
                        threshold: float = 2.0) -> List[int]:
        """
        Detect anomalies in compressed data
        
        Args:
            compressed_data: Compressed segment
            compression_method: Compression method
            threshold: Z-score threshold
            
        Returns:
            List of anomaly indices
        """
        if compression_method == 'fourier':
            return self._detect_anomalies_fourier(compressed_data, threshold)
        
        elif compression_method == 'paa':
            return self._detect_anomalies_paa(compressed_data, threshold)
        
        else:
            return []
    
    def _detect_anomalies_fourier(self, compressed_data: bytes, threshold: float) -> List[int]:
        """Detect anomalies using Fourier representation"""
        compressed = pickle.loads(compressed_data)
        
        # High-frequency components often indicate anomalies
        magnitudes = np.abs(compressed['coeffs'])
        mean_mag = np.mean(magnitudes)
        std_mag = np.std(magnitudes)
        
        z_scores = (magnitudes - mean_mag) / std_mag if std_mag > 0 else np.zeros_like(magnitudes)
        
        anomaly_indices = compressed['indices'][np.abs(z_scores) > threshold]
        return anomaly_indices.tolist()
    
    def _detect_anomalies_paa(self, compressed_data: bytes, threshold: float) -> List[int]:
        """Detect anomalies using PAA representation"""
        paa_array, metadata = pickle.loads(compressed_data)
        
        mean = np.mean(paa_array)
        std = np.std(paa_array)
        
        z_scores = (paa_array - mean) / std if std > 0 else np.zeros_like(paa_array)
        
        anomaly_segments = np.where(np.abs(z_scores) > threshold)[0]
        return anomaly_segments.tolist()
    
    def execute_query(self, query: Dict[str, Any], compressed_segments: List[Tuple]) -> Dict[str, Any]:
        """
        Execute a complex query on compressed data
        
        Args:
            query: Query specification dictionary
            compressed_segments: List of (compressed_data, metadata) tuples
            
        Returns:
            Query results
        """
        query_type = query.get('type')
        
        if query_type == 'range_filter':
            min_val = query['min']
            max_val = query['max']
            results = []
            
            for compressed_data, metadata in compressed_segments:
                method = metadata['compression_method']
                filtered = self.range_filter(compressed_data, method, min_val, max_val)
                if filtered is not None:
                    results.append(filtered)
            
            return {
                'type': 'range_filter',
                'num_segments': len(results),
                'total_points': sum(len(r) for r in results),
                'results': results
            }
        
        elif query_type == 'aggregate':
            agg_func = query['function']
            values = []
            
            for compressed_data, metadata in compressed_segments:
                method = metadata['compression_method']
                val = self.aggregate(compressed_data, method, agg_func)
                if val is not None:
                    values.append(val)
            
            # Combine segment aggregations
            if agg_func == 'mean':
                result = np.mean(values) if values else None
            elif agg_func == 'sum':
                result = np.sum(values) if values else None
            elif agg_func == 'min':
                result = np.min(values) if values else None
            elif agg_func == 'max':
                result = np.max(values) if values else None
            else:
                result = None
            
            return {
                'type': 'aggregate',
                'function': agg_func,
                'result': result,
                'num_segments': len(values)
            }
        
        elif query_type == 'anomaly_detection':
            threshold = query.get('threshold', 2.0)
            all_anomalies = []
            
            for i, (compressed_data, metadata) in enumerate(compressed_segments):
                method = metadata['compression_method']
                anomalies = self.detect_anomalies(compressed_data, method, threshold)
                all_anomalies.extend([(i, a) for a in anomalies])
            
            return {
                'type': 'anomaly_detection',
                'threshold': threshold,
                'num_anomalies': len(all_anomalies),
                'anomalies': all_anomalies
            }
        
        else:
            return {'error': f'Unknown query type: {query_type}'}


def main():
    """Demo query engine"""
    print("=" * 60)
    print("Query Engine Demo")
    print("=" * 60)
    
    from advanced_compression import AdvancedCompressionEngine
    
    # Create sample data
    data = np.random.randn(1000) * 10 + 50
    data[100:110] = 200  # Add anomaly
    
    # Compress with different methods
    engine = AdvancedCompressionEngine()
    query_engine = QueryEngine()
    
    print("\n📊 Compressing sample data...")
    
    methods = ['paa', 'fourier', 'splitdouble']
    compressed_segments = []
    
    for method in methods:
        compressed, metadata = engine.compress(data, method)
        compressed_segments.append((compressed, {'compression_method': method, **metadata}))
        print(f"   {method}: {metadata['compressed_size']} bytes (ratio: {metadata['compression_ratio']:.2f}x)")
    
    # Test range filter
    print("\n🔍 Range Filter Query (40 <= x <= 60):")
    query = {'type': 'range_filter', 'min': 40, 'max': 60}
    result = query_engine.execute_query(query, compressed_segments)
    print(f"   Segments processed: {result['num_segments']}")
    print(f"   Total points matching: {result['total_points']}")
    
    # Test aggregation
    print("\n📈 Aggregation Query (MEAN):")
    query = {'type': 'aggregate', 'function': 'mean'}
    result = query_engine.execute_query(query, compressed_segments)
    print(f"   Result: {result['result']:.2f}")
    print(f"   Actual mean: {np.mean(data):.2f}")
    
    # Test anomaly detection
    print("\n⚠️  Anomaly Detection:")
    query = {'type': 'anomaly_detection', 'threshold': 2.0}
    result = query_engine.execute_query(query, compressed_segments)
    print(f"   Anomalies detected: {result['num_anomalies']}")
    print(f"   Locations: {result['anomalies'][:5]}...")  # Show first 5
    
    print("\n✅ Query engine tests complete!")


if __name__ == "__main__":
    main()
