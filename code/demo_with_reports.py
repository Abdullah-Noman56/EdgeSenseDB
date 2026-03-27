"""
VergeDB Demo with Integrated Reporting
Demonstrates VergeDB features and generates comprehensive reports
"""

import time
import numpy as np
from datetime import datetime, timedelta
from verge_database import VergeDB, AnalyticalTask
from advanced_compression import AdvancedCompressionEngine
from vergedb_reporter import VergeDBReporter


def generate_sensor_data(n_points: int, sensor_type: str = 'temperature') -> np.ndarray:
    """Generate realistic sensor data"""
    t = np.linspace(0, 24, n_points)  # 24 hours
    
    if sensor_type == 'temperature':
        data = 20 + 5 * np.sin(2 * np.pi * t / 24) + np.random.randn(n_points) * 0.5
        anomalies = np.random.choice(n_points, size=5, replace=False)
        data[anomalies] += np.random.randn(5) * 5
    elif sensor_type == 'humidity':
        data = 60 - 10 * np.sin(2 * np.pi * t / 24) + np.random.randn(n_points) * 2
    elif sensor_type == 'pressure':
        data = 1013 + np.random.randn(n_points) * 2
    elif sensor_type == 'cpu':
        data = np.zeros(n_points)
        for i in range(0, n_points, 100):
            burst_len = min(20, n_points - i)
            data[i:i+burst_len] = 80 + np.random.randn(burst_len) * 10
        data += 20 + np.random.randn(n_points) * 5
        data = np.clip(data, 0, 100)
    else:
        data = np.random.randn(n_points) * 10 + 50
    
    return data


def run_demo_with_reports():
    """Run VergeDB demo and generate comprehensive reports"""
    
    print("\n" + "=" * 70)
    print("VergeDB Demo with Integrated Reporting")
    print("=" * 70)
    
    # Create VergeDB instance
    print("\n📊 Initializing VergeDB...")
    db = VergeDB(
        data_dir="data/vergedb_demo",
        compression_threads=3,
        segment_size=512,
        buffer_capacity=50
    )
    
    # Register multiple signals
    signals = [
        ('temperature_01', 'paa', AnalyticalTask.AGGREGATION),
        ('humidity_01', 'fourier', AnalyticalTask.ANOMALY_DETECTION),
        ('pressure_01', 'splitdouble', AnalyticalTask.AGGREGATION),
        ('cpu_usage', 'block_subsample', AnalyticalTask.CLASSIFICATION),
    ]
    
    print("\n📝 Registering signals...")
    for signal_id, method, task in signals:
        db.register_signal(signal_id, compression_method=method, analytical_task=task)
        print(f"   ✓ {signal_id}: {method} (task: {task.value})")
    
    # Start compression threads
    db.start_compression()
    
    # Ingest data
    print("\n💾 Ingesting data...")
    n_points = 2000
    start_time = datetime.now()
    
    for signal_id, _, task in signals:
        sensor_type = signal_id.split('_')[0]
        data = generate_sensor_data(n_points, sensor_type)
        
        batch = []
        for i, value in enumerate(data):
            timestamp = start_time + timedelta(seconds=i)
            batch.append((timestamp, float(value)))
        
        db.ingest_batch(signal_id, batch)
        print(f"   ✓ Ingested {n_points} points for {signal_id}")
    
    # Let compression threads work
    print("\n⏳ Processing and compressing data...")
    time.sleep(3)
    
    # Get statistics
    stats = db.get_statistics()
    
    print(f"\n📊 Database Statistics:")
    print(f"   Total signals: {stats['total_signals']}")
    print(f"   Total data points: {stats['total_data_points_ingested']:,}")
    print(f"   Segments compressed: {stats['total_segments_compressed']}")
    print(f"   Ingestion rate: {stats['ingestion_rate_per_sec']:.2f} points/sec")
    
    # Test compression methods
    print("\n🔧 Testing all compression methods...")
    engine = AdvancedCompressionEngine()
    test_data = generate_sensor_data(5000, 'temperature')
    
    methods = ['gzip', 'gorilla', 'sprintz', 'paa', 'fourier', 
               'block_subsample', 'splitdouble', 'uniform_subsample']
    
    compression_results = {}
    for method in methods:
        try:
            compressed, metadata = engine.compress(test_data, method)
            compression_results[method] = metadata
            print(f"   ✓ {method}: {metadata['compression_ratio']:.2f}x")
        except Exception as e:
            print(f"   ✗ {method}: {str(e)[:50]}")
    
    # Shutdown database
    db.shutdown()
    
    # Generate Reports
    print("\n" + "=" * 70)
    print("📊 Generating Performance Reports")
    print("=" * 70)
    
    reporter = VergeDBReporter()
    
    # Add compression results to stats
    if compression_results:
        reporter.generate_compression_comparison(compression_results)
    
    # Generate storage efficiency chart
    if 'total_original_bytes' in stats and 'total_compressed_bytes' in stats:
        reporter.generate_storage_efficiency(
            stats['total_original_bytes'],
            stats['total_compressed_bytes']
        )
    
    # Generate comprehensive summary
    reporter.generate_summary_report(stats, compression_results)
    
    print("\n" + "=" * 70)
    print("✅ Demo Completed Successfully!")
    print("=" * 70)
    
    print("\n📁 Generated Files:")
    print("   • results/compression_comparison.png - Compression methods comparison")
    print("   • results/storage_efficiency.png - Storage efficiency visualization")
    print("   • results/summary_report.txt - Detailed text report")
    print("\n🎯 Next Steps:")
    print("   • Review the generated charts and report")
    print("   • Modify parameters in the demo to test different scenarios")
    print("   • Integrate the reporter into your own VergeDB applications")


if __name__ == "__main__":
    run_demo_with_reports()
