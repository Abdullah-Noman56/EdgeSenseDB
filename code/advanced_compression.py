"""
Advanced Compression Engine for VergeDB
Implements compression methods from the VergeDB paper including:
- Gorilla (XOR-based float encoding)
- Sprintz (predictive encoding)
- PAA (Piecewise Aggregate Approximation)
- Fourier Transform
- Block Subsampling
- SplitDouble (precision-aware compression)
- Snappy compression
"""

import struct
import gzip
import math
import pickle
from typing import Tuple, Dict, Any, List
import numpy as np

try:
    import snappy
    HAS_SNAPPY = True
except ImportError:
    HAS_SNAPPY = False


class GorillaCompressor:
    """
    Gorilla compression for float time-series
    Based on Facebook's Gorilla TSDB paper
    Uses XOR encoding for floats
    """
    
    @staticmethod
    def compress(data: np.ndarray) -> bytes:
        """Compress float array using Gorilla XOR encoding"""
        if len(data) == 0:
            return b''
        
        result = []
        
        # Store first value as-is
        prev_value = data[0]
        result.append(struct.pack('d', prev_value))
        
        prev_xor = 0
        prev_leading_zeros = 0
        prev_trailing_zeros = 0
        
        for value in data[1:]:
            # Convert to uint64 for XOR
            value_bits = struct.unpack('Q', struct.pack('d', value))[0]
            prev_bits = struct.unpack('Q', struct.pack('d', prev_value))[0]
            
            xor = value_bits ^ prev_bits
            
            if xor == 0:
                # Value unchanged, store control bit
                result.append(b'\x00')
            else:
                # Count leading and trailing zeros
                leading_zeros = 64 - xor.bit_length()
                trailing_zeros = (xor & -xor).bit_length() - 1 if xor != 0 else 0
                
                # Store XOR with metadata
                result.append(struct.pack('QBB', xor, leading_zeros, trailing_zeros))
                
                prev_leading_zeros = leading_zeros
                prev_trailing_zeros = trailing_zeros
            
            prev_value = value
            prev_xor = xor
        
        return b''.join(result)
    
    @staticmethod
    def decompress(data: bytes) -> np.ndarray:
        """Decompress Gorilla encoded data"""
        if not data:
            return np.array([])
        
        result = []
        offset = 0
        
        # Read first value
        first_value = struct.unpack('d', data[offset:offset+8])[0]
        result.append(first_value)
        offset += 8
        
        prev_value = first_value
        
        while offset < len(data):
            if data[offset] == 0:
                # Value unchanged
                result.append(prev_value)
                offset += 1
            else:
                # Read XOR and metadata
                xor, leading, trailing = struct.unpack('QBB', data[offset:offset+10])
                offset += 10
                
                # Reconstruct value
                prev_bits = struct.unpack('Q', struct.pack('d', prev_value))[0]
                value_bits = prev_bits ^ xor
                value = struct.unpack('d', struct.pack('Q', value_bits))[0]
                
                result.append(value)
                prev_value = value
        
        return np.array(result)


class SprintzCompressor:
    """
    Sprintz compression for time-series
    Uses predictive encoding with bit-packing
    """
    
    @staticmethod
    def _predict(prev: float, prev2: float) -> float:
        """Simple linear predictor"""
        return 2 * prev - prev2
    
    @staticmethod
    def compress(data: np.ndarray) -> bytes:
        """Compress using Sprintz predictive encoding"""
        if len(data) < 2:
            return pickle.dumps(data)
        
        # Store first two values
        result = [struct.pack('dd', data[0], data[1])]
        
        # Predict and store deltas
        for i in range(2, len(data)):
            predicted = SprintzCompressor._predict(data[i-1], data[i-2])
            delta = data[i] - predicted
            
            # Quantize delta to reduce size
            quantized = int(delta * 100)  # 2 decimal precision
            result.append(struct.pack('i', quantized))
        
        return b''.join(result)
    
    @staticmethod
    def decompress(data: bytes) -> np.ndarray:
        """Decompress Sprintz encoded data"""
        if len(data) < 16:
            return pickle.loads(data)
        
        result = []
        
        # Read first two values
        val0, val1 = struct.unpack('dd', data[0:16])
        result.extend([val0, val1])
        
        offset = 16
        while offset < len(data):
            quantized = struct.unpack('i', data[offset:offset+4])[0]
            offset += 4
            
            delta = quantized / 100.0
            predicted = SprintzCompressor._predict(result[-1], result[-2])
            value = predicted + delta
            
            result.append(value)
        
        return np.array(result)


class PAACompressor:
    """
    Piecewise Aggregate Approximation
    Reduces dimensionality by averaging segments
    """
    
    @staticmethod
    def compress(data: np.ndarray, segments: int = None) -> Tuple[bytes, Dict]:
        """Compress using PAA"""
        if segments is None:
            segments = max(1, len(data) // 4)
        
        n = len(data)
        segment_size = n / segments
        
        paa_repr = []
        for i in range(segments):
            start = int(i * segment_size)
            end = int((i + 1) * segment_size)
            segment_mean = np.mean(data[start:end])
            paa_repr.append(segment_mean)
        
        paa_array = np.array(paa_repr, dtype=np.float32)
        
        metadata = {
            'original_length': n,
            'segments': segments,
            'segment_size': segment_size
        }
        
        return pickle.dumps((paa_array, metadata)), metadata
    
    @staticmethod
    def decompress(data: bytes) -> np.ndarray:
        """Decompress PAA representation"""
        paa_array, metadata = pickle.loads(data)
        
        # Reconstruct by repeating each segment value
        n = metadata['original_length']
        segments = metadata['segments']
        segment_size = metadata['segment_size']
        
        reconstructed = []
        for i, val in enumerate(paa_array):
            start = int(i * segment_size)
            end = int((i + 1) * segment_size)
            reconstructed.extend([val] * (end - start))
        
        return np.array(reconstructed[:n])


class FourierCompressor:
    """
    Discrete Fourier Transform compression
    Keeps only top-k frequency components
    """
    
    @staticmethod
    def compress(data: np.ndarray, keep_ratio: float = 0.1) -> Tuple[bytes, Dict]:
        """Compress using DFT, keeping only top frequencies"""
        # Compute FFT
        fft_coeffs = np.fft.fft(data)
        
        # Keep only top-k coefficients by magnitude
        magnitudes = np.abs(fft_coeffs)
        k = max(1, int(len(data) * keep_ratio))
        
        # Get indices of top-k coefficients
        top_indices = np.argsort(magnitudes)[-k:]
        
        # Store only non-zero coefficients
        compressed = {
            'length': len(data),
            'coeffs': fft_coeffs[top_indices].astype(np.complex64),
            'indices': top_indices
        }
        
        metadata = {
            'original_length': len(data),
            'kept_coefficients': k,
            'keep_ratio': keep_ratio
        }
        
        return pickle.dumps(compressed), metadata
    
    @staticmethod
    def decompress(data: bytes) -> np.ndarray:
        """Decompress Fourier representation"""
        compressed = pickle.loads(data)
        
        # Reconstruct full FFT array
        n = compressed['length']
        fft_coeffs = np.zeros(n, dtype=np.complex128)
        fft_coeffs[compressed['indices']] = compressed['coeffs']
        
        # Inverse FFT
        reconstructed = np.fft.ifft(fft_coeffs)
        return np.real(reconstructed)


class BlockSubsampler:
    """
    Block Subsampling as described in VergeDB paper
    Samples contiguous blocks to preserve local trends
    """
    
    @staticmethod
    def subsample(data: np.ndarray, block_size: int = 10, sample_rate: float = 0.1) -> Tuple[bytes, Dict]:
        """
        Subsample data in blocks
        
        Args:
            data: Input time series
            block_size: Size of each block
            sample_rate: Fraction of blocks to keep
        """
        n = len(data)
        num_blocks = n // block_size
        num_keep = max(1, int(num_blocks * sample_rate))
        
        # Randomly select blocks to keep (could be made adaptive)
        np.random.seed(42)
        selected_blocks = sorted(np.random.choice(num_blocks, num_keep, replace=False))
        
        sampled_data = []
        block_info = []
        
        for block_idx in selected_blocks:
            start = block_idx * block_size
            end = min(start + block_size, n)
            block = data[start:end]
            sampled_data.extend(block)
            block_info.append((start, end))
        
        compressed = {
            'data': np.array(sampled_data, dtype=np.float32),
            'blocks': block_info,
            'original_length': n,
            'block_size': block_size
        }
        
        metadata = {
            'original_length': n,
            'sampled_length': len(sampled_data),
            'sample_rate': len(sampled_data) / n,
            'num_blocks': num_keep
        }
        
        return pickle.dumps(compressed), metadata
    
    @staticmethod
    def reconstruct(data: bytes) -> np.ndarray:
        """Reconstruct from block subsampled data (with interpolation)"""
        compressed = pickle.loads(data)
        
        n = compressed['original_length']
        sampled = compressed['data']
        blocks = compressed['blocks']
        
        # Simple reconstruction: interpolate between blocks
        reconstructed = np.zeros(n)
        
        offset = 0
        for start, end in blocks:
            block_len = end - start
            reconstructed[start:end] = sampled[offset:offset+block_len]
            offset += block_len
        
        # Fill gaps with interpolation
        mask = reconstructed != 0
        if np.any(mask):
            indices = np.arange(n)
            reconstructed[~mask] = np.interp(indices[~mask], indices[mask], reconstructed[mask])
        
        return reconstructed


class SplitDoubleCompressor:
    """
    SplitDouble: Precision-aware compression for bounded range data
    Splits float into integer and fractional parts
    """
    
    @staticmethod
    def compress(data: np.ndarray, precision: int = 2) -> Tuple[bytes, Dict]:
        """
        Compress by splitting into integer and decimal parts
        
        Args:
            data: Input data
            precision: Number of decimal places to preserve
        """
        multiplier = 10 ** precision
        
        # Scale and split
        scaled = data * multiplier
        integers = scaled.astype(np.int32)
        
        # Delta encoding on integers
        deltas = np.diff(integers, prepend=integers[0])
        
        # Bit-pack deltas
        compressed = {
            'first': integers[0],
            'deltas': deltas.astype(np.int16),  # Assuming small deltas
            'precision': precision
        }
        
        metadata = {
            'original_length': len(data),
            'precision': precision,
            'range': (float(np.min(data)), float(np.max(data)))
        }
        
        return pickle.dumps(compressed), metadata
    
    @staticmethod
    def decompress(data: bytes) -> np.ndarray:
        """Decompress SplitDouble representation"""
        compressed = pickle.loads(data)
        
        # Reconstruct integers from deltas
        deltas = compressed['deltas']
        first = compressed['first']
        precision = compressed['precision']
        
        integers = np.cumsum(deltas)
        
        # Convert back to floats
        multiplier = 10 ** precision
        result = integers.astype(np.float64) / multiplier
        
        return result


class AdvancedCompressionEngine:
    """
    Advanced compression engine implementing VergeDB compression methods
    """
    
    def __init__(self):
        self.methods = {
            'gzip': self._gzip,
            'snappy': self._snappy,
            'gorilla': self._gorilla,
            'sprintz': self._sprintz,
            'paa': self._paa,
            'fourier': self._fourier,
            'block_subsample': self._block_subsample,
            'splitdouble': self._splitdouble,
            'uniform_subsample': self._uniform_subsample
        }
    
    def compress(self, data: np.ndarray, method: str = 'paa') -> Tuple[bytes, Dict[str, Any]]:
        """
        Compress data using specified method
        
        Args:
            data: NumPy array of time-series data
            method: Compression method name
            
        Returns:
            Tuple of (compressed_bytes, metadata)
        """
        if method not in self.methods:
            raise ValueError(f"Unknown compression method: {method}")
        
        return self.methods[method](data)
    
    def _gzip(self, data: np.ndarray) -> Tuple[bytes, Dict]:
        """Gzip compression"""
        serialized = pickle.dumps(data)
        compressed = gzip.compress(serialized)
        
        compression_ratio = len(serialized) / len(compressed) if len(compressed) > 0 else 0
        space_saved = ((len(serialized) - len(compressed)) / len(serialized) * 100) if len(serialized) > 0 else 0
        
        metadata = {
            'method': 'gzip',
            'original_size': len(serialized),
            'compressed_size': len(compressed),
            'compression_ratio': compression_ratio,
            'space_saved_percent': space_saved
        }
        
        return compressed, metadata
    
    def _snappy(self, data: np.ndarray) -> Tuple[bytes, Dict]:
        """Snappy compression"""
        if not HAS_SNAPPY:
            print("⚠️  Snappy not available, falling back to gzip")
            return self._gzip(data)
        
        serialized = pickle.dumps(data)
        compressed = snappy.compress(serialized)
        
        compression_ratio = len(serialized) / len(compressed) if len(compressed) > 0 else 0
        space_saved = ((len(serialized) - len(compressed)) / len(serialized) * 100) if len(serialized) > 0 else 0
        
        metadata = {
            'method': 'snappy',
            'original_size': len(serialized),
            'compressed_size': len(compressed),
            'compression_ratio': compression_ratio,
            'space_saved_percent': space_saved
        }
        
        return compressed, metadata
    
    def _gorilla(self, data: np.ndarray) -> Tuple[bytes, Dict]:
        """Gorilla XOR encoding"""
        compressed = GorillaCompressor.compress(data)
        
        compression_ratio = data.nbytes / len(compressed) if len(compressed) > 0 else 0
        space_saved = ((data.nbytes - len(compressed)) / data.nbytes * 100) if data.nbytes > 0 else 0
        
        metadata = {
            'method': 'gorilla',
            'original_size': data.nbytes,
            'compressed_size': len(compressed),
            'compression_ratio': compression_ratio,
            'space_saved_percent': space_saved
        }
        
        return compressed, metadata
    
    def _sprintz(self, data: np.ndarray) -> Tuple[bytes, Dict]:
        """Sprintz predictive encoding"""
        compressed = SprintzCompressor.compress(data)
        
        compression_ratio = data.nbytes / len(compressed) if len(compressed) > 0 else 0
        space_saved = ((data.nbytes - len(compressed)) / data.nbytes * 100) if data.nbytes > 0 else 0
        
        metadata = {
            'method': 'sprintz',
            'original_size': data.nbytes,
            'compressed_size': len(compressed),
            'compression_ratio': compression_ratio,
            'space_saved_percent': space_saved
        }
        
        return compressed, metadata
    
    def _paa(self, data: np.ndarray) -> Tuple[bytes, Dict]:
        """PAA compression"""
        segments = max(1, len(data) // 4)
        compressed, paa_metadata = PAACompressor.compress(data, segments)
        
        compression_ratio = data.nbytes / len(compressed) if len(compressed) > 0 else 0
        space_saved = ((data.nbytes - len(compressed)) / data.nbytes * 100) if data.nbytes > 0 else 0
        
        metadata = {
            'method': 'paa',
            'original_size': data.nbytes,
            'compressed_size': len(compressed),
            'compression_ratio': compression_ratio,
            'space_saved_percent': space_saved,
            **paa_metadata
        }
        
        return compressed, metadata
    
    def _fourier(self, data: np.ndarray) -> Tuple[bytes, Dict]:
        """Fourier transform compression"""
        compressed, fourier_metadata = FourierCompressor.compress(data, keep_ratio=0.1)
        
        compression_ratio = data.nbytes / len(compressed) if len(compressed) > 0 else 0
        space_saved = ((data.nbytes - len(compressed)) / data.nbytes * 100) if data.nbytes > 0 else 0
        
        metadata = {
            'method': 'fourier',
            'original_size': data.nbytes,
            'compressed_size': len(compressed),
            'compression_ratio': compression_ratio,
            'space_saved_percent': space_saved,
            **fourier_metadata
        }
        
        return compressed, metadata
    
    def _block_subsample(self, data: np.ndarray) -> Tuple[bytes, Dict]:
        """Block subsampling"""
        compressed, subsample_metadata = BlockSubsampler.subsample(data, block_size=10, sample_rate=0.2)
        
        compression_ratio = data.nbytes / len(compressed) if len(compressed) > 0 else 0
        space_saved = ((data.nbytes - len(compressed)) / data.nbytes * 100) if data.nbytes > 0 else 0
        
        metadata = {
            'method': 'block_subsample',
            'original_size': data.nbytes,
            'compressed_size': len(compressed),
            'compression_ratio': compression_ratio,
            'space_saved_percent': space_saved,
            **subsample_metadata
        }
        
        return compressed, metadata
    
    def _splitdouble(self, data: np.ndarray) -> Tuple[bytes, Dict]:
        """SplitDouble compression"""
        compressed, split_metadata = SplitDoubleCompressor.compress(data, precision=2)
        
        compression_ratio = data.nbytes / len(compressed) if len(compressed) > 0 else 0
        space_saved = ((data.nbytes - len(compressed)) / data.nbytes * 100) if data.nbytes > 0 else 0
        
        metadata = {
            'method': 'splitdouble',
            'original_size': data.nbytes,
            'compressed_size': len(compressed),
            'compression_ratio': compression_ratio,
            'space_saved_percent': space_saved,
            **split_metadata
        }
        
        return compressed, metadata
    
    def _uniform_subsample(self, data: np.ndarray, rate: float = 0.1) -> Tuple[bytes, Dict]:
        """Uniform subsampling"""
        n = len(data)
        k = max(1, int(n * rate))
        indices = np.linspace(0, n-1, k, dtype=int)
        sampled = data[indices]
        
        compressed = pickle.dumps({'data': sampled, 'indices': indices, 'length': n})
        
        compression_ratio = data.nbytes / len(compressed) if len(compressed) > 0 else 0
        space_saved = ((data.nbytes - len(compressed)) / data.nbytes * 100) if data.nbytes > 0 else 0
        
        metadata = {
            'method': 'uniform_subsample',
            'original_size': data.nbytes,
            'compressed_size': len(compressed),
            'compression_ratio': compression_ratio,
            'space_saved_percent': space_saved,
            'sample_rate': rate
        }
        
        return compressed, metadata
    
    def decompress(self, data: bytes, method: str) -> np.ndarray:
        """Decompress data"""
        decompressors = {
            'gzip': lambda d: pickle.loads(gzip.decompress(d)),
            'snappy': lambda d: pickle.loads(snappy.decompress(d)) if HAS_SNAPPY else pickle.loads(gzip.decompress(d)),
            'gorilla': GorillaCompressor.decompress,
            'sprintz': SprintzCompressor.decompress,
            'paa': PAACompressor.decompress,
            'fourier': FourierCompressor.decompress,
            'block_subsample': BlockSubsampler.reconstruct,
            'splitdouble': SplitDoubleCompressor.decompress
        }
        
        if method not in decompressors:
            raise ValueError(f"Unknown decompression method: {method}")
        
        return decompressors[method](data)
