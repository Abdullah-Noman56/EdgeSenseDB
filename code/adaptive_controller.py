"""
Adaptive Controller for VergeDB
Dynamically selects compression methods based on:
1. Downstream analytical task
2. System resources (CPU, memory, storage)
3. Data characteristics (rate, distribution, cardinality)
"""

import psutil
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ResourceStatus(Enum):
    """System resource status"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class SystemResources:
    """Current system resource availability"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bandwidth: Optional[float] = None
    
    def get_cpu_status(self) -> ResourceStatus:
        if self.cpu_percent < 30:
            return ResourceStatus.LOW
        elif self.cpu_percent < 70:
            return ResourceStatus.MEDIUM
        return ResourceStatus.HIGH
    
    def get_memory_status(self) -> ResourceStatus:
        if self.memory_percent < 50:
            return ResourceStatus.LOW
        elif self.memory_percent < 80:
            return ResourceStatus.MEDIUM
        return ResourceStatus.HIGH


@dataclass
class DataCharacteristics:
    """Characteristics of incoming data"""
    ingestion_rate: float  # points per second
    cardinality: int  # unique values
    range_min: float
    range_max: float
    variance: float
    is_sorted: bool
    has_trends: bool


class CompressionStrategy:
    """Compression strategy recommendation"""
    
    def __init__(self, method: str, priority: int, reason: str):
        self.method = method
        self.priority = priority
        self.reason = reason
    
    def __repr__(self):
        return f"CompressionStrategy({self.method}, priority={self.priority}, reason='{self.reason}')"


class AdaptiveController:
    """
    Adaptive controller that selects optimal compression strategy
    based on task requirements, resources, and data characteristics
    """
    
    def __init__(self, monitoring_interval: float = 5.0):
        """
        Initialize adaptive controller
        
        Args:
            monitoring_interval: Seconds between resource checks
        """
        self.monitoring_interval = monitoring_interval
        self.last_check_time = 0
        self.cached_resources = None
        
        # Compression method properties
        self.method_properties = {
            'gzip': {
                'cpu_cost': 'high',
                'compression_ratio': 'high',
                'query_support': 'none',
                'task_suitability': ['aggregation', 'general']
            },
            'snappy': {
                'cpu_cost': 'low',
                'compression_ratio': 'medium',
                'query_support': 'none',
                'task_suitability': ['general', 'low_latency']
            },
            'gorilla': {
                'cpu_cost': 'medium',
                'compression_ratio': 'high',
                'query_support': 'partial',
                'task_suitability': ['aggregation', 'float_data']
            },
            'sprintz': {
                'cpu_cost': 'medium',
                'compression_ratio': 'high',
                'query_support': 'partial',
                'task_suitability': ['forecasting', 'trending_data']
            },
            'paa': {
                'cpu_cost': 'low',
                'compression_ratio': 'medium',
                'query_support': 'full',
                'task_suitability': ['aggregation', 'visualization']
            },
            'fourier': {
                'cpu_cost': 'medium',
                'compression_ratio': 'high',
                'query_support': 'full',
                'task_suitability': ['anomaly_detection', 'frequency_analysis']
            },
            'block_subsample': {
                'cpu_cost': 'low',
                'compression_ratio': 'very_high',
                'query_support': 'full',
                'task_suitability': ['forecasting', 'trend_detection', 'classification']
            },
            'splitdouble': {
                'cpu_cost': 'low',
                'compression_ratio': 'high',
                'query_support': 'full',
                'task_suitability': ['aggregation', 'filtering', 'bounded_data']
            },
            'uniform_subsample': {
                'cpu_cost': 'very_low',
                'compression_ratio': 'very_high',
                'query_support': 'partial',
                'task_suitability': ['aggregation', 'counting']
            }
        }
    
    def get_system_resources(self) -> SystemResources:
        """Get current system resource usage"""
        current_time = time.time()
        
        # Cache resource checks to avoid overhead
        if (self.cached_resources is not None and 
            current_time - self.last_check_time < self.monitoring_interval):
            return self.cached_resources
        
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        
        resources = SystemResources(
            cpu_percent=cpu,
            memory_percent=memory,
            disk_percent=disk
        )
        
        self.cached_resources = resources
        self.last_check_time = current_time
        
        return resources
    
    def analyze_data_characteristics(self, data: np.ndarray) -> DataCharacteristics:
        """Analyze characteristics of incoming data"""
        if len(data) == 0:
            return DataCharacteristics(0, 0, 0, 0, 0, False, False)
        
        unique_values = len(np.unique(data))
        is_sorted = np.all(np.diff(data) >= 0) or np.all(np.diff(data) <= 0)
        
        # Simple trend detection
        if len(data) > 2:
            diffs = np.diff(data)
            has_trends = np.std(diffs) > 0.01 * np.std(data)
        else:
            has_trends = False
        
        return DataCharacteristics(
            ingestion_rate=0,  # Would be calculated from actual ingestion
            cardinality=unique_values,
            range_min=float(np.min(data)),
            range_max=float(np.max(data)),
            variance=float(np.var(data)),
            is_sorted=is_sorted,
            has_trends=has_trends
        )
    
    def select_compression_method(self, 
                                  task: Optional[str] = None,
                                  data_chars: Optional[DataCharacteristics] = None,
                                  resources: Optional[SystemResources] = None) -> CompressionStrategy:
        """
        Select optimal compression method
        
        Args:
            task: Downstream analytical task
            data_chars: Data characteristics
            resources: System resources (if None, will check automatically)
            
        Returns:
            CompressionStrategy with selected method and reasoning
        """
        if resources is None:
            resources = self.get_system_resources()
        
        strategies = []
        
        # Task-based selection
        if task:
            task_map = {
                'aggregation': ['paa', 'splitdouble', 'gorilla'],
                'classification': ['block_subsample', 'fourier'],
                'clustering': ['fourier', 'paa'],
                'anomaly_detection': ['fourier', 'paa'],
                'similarity_search': ['fourier', 'block_subsample'],
                'forecasting': ['block_subsample', 'sprintz']
            }
            
            if task in task_map:
                for method in task_map[task]:
                    strategies.append(
                        CompressionStrategy(method, 10, f"Optimized for {task}")
                    )
        
        # Resource-based selection
        cpu_status = resources.get_cpu_status()
        memory_status = resources.get_memory_status()
        
        if cpu_status == ResourceStatus.HIGH:
            # Low CPU methods
            strategies.append(
                CompressionStrategy('uniform_subsample', 8, "Low CPU usage")
            )
            strategies.append(
                CompressionStrategy('paa', 7, "Medium CPU usage")
            )
        elif cpu_status == ResourceStatus.LOW:
            # Can afford expensive methods
            strategies.append(
                CompressionStrategy('gzip', 6, "High compression ratio")
            )
            strategies.append(
                CompressionStrategy('fourier', 6, "Good for analytics")
            )
        
        if memory_status == ResourceStatus.HIGH:
            # Aggressive compression
            strategies.append(
                CompressionStrategy('gzip', 9, "Save memory")
            )
        
        # Data-based selection
        if data_chars:
            if data_chars.cardinality < 100:
                strategies.append(
                    CompressionStrategy('splitdouble', 8, "Low cardinality")
                )
            
            if data_chars.has_trends:
                strategies.append(
                    CompressionStrategy('block_subsample', 7, "Has trends")
                )
                strategies.append(
                    CompressionStrategy('sprintz', 6, "Good for trending data")
                )
            
            if data_chars.is_sorted:
                strategies.append(
                    CompressionStrategy('gorilla', 7, "Sorted data")
                )
                strategies.append(
                    CompressionStrategy('sprintz', 7, "Sorted data")
                )
        
        # Default fallback
        if not strategies:
            strategies.append(
                CompressionStrategy('paa', 5, "Default choice")
            )
        
        # Select best strategy by priority
        best_strategy = max(strategies, key=lambda s: s.priority)
        
        return best_strategy
    
    def should_switch_compression(self, current_method: str, 
                                 current_performance: Dict[str, float],
                                 signal_id: str) -> Optional[str]:
        """
        Determine if compression method should be switched
        
        Args:
            current_method: Current compression method
            current_performance: Performance metrics
            signal_id: Signal identifier
            
        Returns:
            New compression method or None if no switch needed
        """
        resources = self.get_system_resources()
        
        # Check if current method is struggling
        compression_ratio = current_performance.get('compression_ratio', 1.0)
        compression_time = current_performance.get('compression_time', 0)
        
        # If compression ratio is poor (<1.5x), switch to more aggressive
        if compression_ratio < 1.5:
            if current_method in ['uniform_subsample', 'paa']:
                return 'gzip'
            elif current_method == 'splitdouble':
                return 'fourier'
        
        # If CPU is high and using expensive method, downgrade
        if resources.get_cpu_status() == ResourceStatus.HIGH:
            if current_method in ['gzip', 'fourier']:
                return 'paa'
        
        # If CPU is low and using cheap method, upgrade
        if resources.get_cpu_status() == ResourceStatus.LOW:
            if current_method in ['uniform_subsample', 'paa']:
                return 'fourier'
        
        return None
    
    def get_recommendations(self, task: Optional[str] = None) -> List[CompressionStrategy]:
        """Get top 3 compression method recommendations"""
        resources = self.get_system_resources()
        
        strategies = []
        
        for method in self.method_properties.keys():
            props = self.method_properties[method]
            priority = 5  # base priority
            reasons = []
            
            # Task matching
            if task and task in props['task_suitability']:
                priority += 3
                reasons.append(f"suitable for {task}")
            
            # Resource matching
            cpu_status = resources.get_cpu_status()
            if cpu_status == ResourceStatus.HIGH and props['cpu_cost'] in ['low', 'very_low']:
                priority += 2
                reasons.append("low CPU cost")
            elif cpu_status == ResourceStatus.LOW and props['cpu_cost'] == 'high':
                priority += 1
                reasons.append("can afford high CPU")
            
            # Query support
            if props['query_support'] == 'full':
                priority += 1
                reasons.append("full query support")
            
            reason = ", ".join(reasons) if reasons else "general purpose"
            strategies.append(CompressionStrategy(method, priority, reason))
        
        # Return top 3
        return sorted(strategies, key=lambda s: s.priority, reverse=True)[:3]
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report"""
        resources = self.get_system_resources()
        
        report = {
            'timestamp': time.time(),
            'system_resources': {
                'cpu_percent': resources.cpu_percent,
                'cpu_status': resources.get_cpu_status().value,
                'memory_percent': resources.memory_percent,
                'memory_status': resources.get_memory_status().value,
                'disk_percent': resources.disk_percent
            },
            'recommendations': {}
        }
        
        # Get recommendations for each task type
        for task in ['aggregation', 'classification', 'forecasting', 'anomaly_detection']:
            top_methods = self.get_recommendations(task)
            report['recommendations'][task] = [
                {'method': s.method, 'priority': s.priority, 'reason': s.reason}
                for s in top_methods
            ]
        
        return report


def main():
    """Demo adaptive controller"""
    print("=" * 60)
    print("Adaptive Controller Demo")
    print("=" * 60)
    
    controller = AdaptiveController()
    
    # Check system resources
    resources = controller.get_system_resources()
    print(f"\n📊 System Resources:")
    print(f"   CPU: {resources.cpu_percent:.1f}% ({resources.get_cpu_status().value})")
    print(f"   Memory: {resources.memory_percent:.1f}% ({resources.get_memory_status().value})")
    print(f"   Disk: {resources.disk_percent:.1f}%")
    
    # Generate sample data
    print(f"\n📈 Analyzing sample data...")
    data = np.random.randn(1000) + np.linspace(0, 10, 1000)  # Trending data
    data_chars = controller.analyze_data_characteristics(data)
    print(f"   Cardinality: {data_chars.cardinality}")
    print(f"   Range: [{data_chars.range_min:.2f}, {data_chars.range_max:.2f}]")
    print(f"   Has trends: {data_chars.has_trends}")
    print(f"   Is sorted: {data_chars.is_sorted}")
    
    # Get recommendations for different tasks
    print(f"\n🎯 Compression Recommendations:")
    
    for task in ['aggregation', 'classification', 'forecasting', 'anomaly_detection']:
        print(f"\n   Task: {task.upper()}")
        strategy = controller.select_compression_method(task, data_chars, resources)
        print(f"   → {strategy.method} (priority: {strategy.priority})")
        print(f"     Reason: {strategy.reason}")
    
    # Generate full report
    print(f"\n📄 Generating full system report...")
    report = controller.generate_report()
    
    print(f"\n   Top recommendations by task:")
    for task, methods in report['recommendations'].items():
        print(f"\n   {task}:")
        for m in methods:
            print(f"     - {m['method']} (priority: {m['priority']}): {m['reason']}")


if __name__ == "__main__":
    main()
