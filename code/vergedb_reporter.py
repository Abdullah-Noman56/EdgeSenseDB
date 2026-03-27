"""
VergeDB Performance Reporter
Generates visualization charts and summary reports for VergeDB implementation

USAGE EXAMPLE:
-------------
from vergedb_reporter import VergeDBReporter

# Get statistics from your VergeDB instance
stats = db.get_statistics()

# Create reporter
reporter = VergeDBReporter(output_dir="results")

# Generate all reports at once
reporter.generate_all_reports(stats, compression_results)

# Or generate individual reports
reporter.generate_compression_comparison(compression_results)
reporter.generate_storage_efficiency(original_size, compressed_size)
reporter.generate_summary_report(stats)

OUTPUT FILES:
------------
- compression_comparison.png: Bar charts comparing all compression methods
- storage_efficiency.png: Pie and bar charts showing storage savings
- summary_report.txt: Comprehensive text report with all metrics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class VergeDBReporter:
    """Performance reporting and visualization for VergeDB"""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize the reporter
        
        Args:
            output_dir: Directory to save reports and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for better-looking plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def generate_compression_comparison(
        self, 
        compression_results: Dict[str, Dict[str, Any]],
        save_path: str = None
    ):
        """
        Generate compression comparison visualization
        
        Args:
            compression_results: Dictionary with compression method as key and metrics as values
                Expected format: {
                    'method_name': {
                        'compressed_size': int,
                        'compression_ratio': float,
                        'space_saved_percent': float,
                        'compression_time': float (optional)
                    }
                }
            save_path: Custom save path (default: results/compression_comparison.png)
        """
        if save_path is None:
            save_path = self.output_dir / "compression_comparison.png"
        
        # Extract data
        methods = []
        ratios = []
        space_saved = []
        
        for method, data in compression_results.items():
            if 'error' not in data:
                methods.append(method.upper())
                ratios.append(data.get('compression_ratio', 0))
                space_saved.append(data.get('space_saved_percent', 0))
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Compression Ratio
        colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
        bars1 = ax1.bar(methods, ratios, color=colors, edgecolor='black', linewidth=1.2)
        ax1.set_ylabel('Compression Ratio (x)', fontsize=12, fontweight='bold')
        ax1.set_title('Compression Ratio by Method', fontsize=14, fontweight='bold', pad=20)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, ratio in zip(bars1, ratios):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ratio:.2f}x',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot 2: Space Saved Percentage
        bars2 = ax2.bar(methods, space_saved, color=colors, edgecolor='black', linewidth=1.2)
        ax2.set_ylabel('Space Saved (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Storage Efficiency by Method', fontsize=14, fontweight='bold', pad=20)
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim([0, 100])
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, saved in zip(bars2, space_saved):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{saved:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Compression comparison chart saved to: {save_path}")
    
    def generate_storage_efficiency(
        self,
        original_size: int,
        compressed_size: int,
        save_path: str = None
    ):
        """
        Generate storage efficiency visualization (pie chart and bar chart)
        
        Args:
            original_size: Original data size in bytes
            compressed_size: Compressed data size in bytes
            save_path: Custom save path (default: results/storage_efficiency.png)
        """
        if save_path is None:
            save_path = self.output_dir / "storage_efficiency.png"
        
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        space_saved = ((original_size - compressed_size) / original_size) * 100 if original_size > 0 else 0
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(14, 6))
        
        # Plot 1: Pie Chart
        ax1 = plt.subplot(1, 2, 1)
        sizes = [compressed_size, original_size - compressed_size]
        labels = ['Compressed\nStorage', 'Space\nSaved']
        colors = ['#ff7f0e', '#2ca02c']
        explode = (0.05, 0.05)
        
        wedges, texts, autotexts = ax1.pie(
            sizes, 
            explode=explode, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%',
            shadow=True, 
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        
        ax1.set_title('Storage Distribution', fontsize=14, fontweight='bold', pad=20)
        
        # Plot 2: Bar Chart Comparison
        ax2 = plt.subplot(1, 2, 2)
        
        # Convert to MB for better readability
        original_mb = original_size / (1024 * 1024)
        compressed_mb = compressed_size / (1024 * 1024)
        
        categories = ['Original', 'Compressed']
        values = [original_mb, compressed_mb]
        colors_bar = ['#1f77b4', '#ff7f0e']
        
        bars = ax2.bar(categories, values, color=colors_bar, edgecolor='black', linewidth=1.5, width=0.6)
        
        ax2.set_ylabel('Storage Size (MB)', fontsize=12, fontweight='bold')
        ax2.set_title('Storage Size Comparison', fontsize=14, fontweight='bold', pad=20)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value, size_bytes in zip(bars, values, [original_size, compressed_size]):
            height = bar.get_height()
            # Show both MB and bytes
            if value >= 1:
                label = f'{value:.2f} MB\n({self._format_bytes(size_bytes)})'
            else:
                label = f'{value:.3f} MB\n({self._format_bytes(size_bytes)})'
            
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    label,
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add compression info as text
        info_text = f'Compression Ratio: {compression_ratio:.2f}x\nSpace Saved: {space_saved:.1f}%'
        ax2.text(0.5, 0.95, info_text,
                transform=ax2.transAxes,
                fontsize=11,
                verticalalignment='top',
                horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Storage efficiency chart saved to: {save_path}")
    
    def generate_summary_report(
        self,
        stats: Dict[str, Any],
        compression_results: Dict[str, Dict[str, Any]] = None,
        save_path: str = None
    ):
        """
        Generate comprehensive text summary report
        
        Args:
            stats: Database statistics from VergeDB.get_statistics()
            compression_results: Optional compression comparison results
            save_path: Custom save path (default: results/summary_report.txt)
        """
        if save_path is None:
            save_path = self.output_dir / "summary_report.txt"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("VERGEDB PERFORMANCE SUMMARY REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Database Statistics
            f.write("DATABASE STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Signals:              {stats.get('total_signals', 0)}\n")
            f.write(f"Total Data Points Ingested: {stats.get('total_data_points_ingested', 0):,}\n")
            f.write(f"Total Segments Compressed:  {stats.get('total_segments_compressed', 0):,}\n")
            f.write(f"Total Segments Flushed:     {stats.get('total_segments_flushed', 0):,}\n")
            f.write(f"Ingestion Rate:             {stats.get('ingestion_rate_per_sec', 0):,.2f} points/sec\n")
            f.write("\n")
            
            # Storage Efficiency
            if 'total_original_bytes' in stats or 'total_compressed_bytes' in stats:
                f.write("STORAGE EFFICIENCY\n")
                f.write("-" * 80 + "\n")
                
                original_bytes = stats.get('total_original_bytes', 0)
                compressed_bytes = stats.get('total_compressed_bytes', 0)
                
                if original_bytes > 0:
                    ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 0
                    space_saved = ((original_bytes - compressed_bytes) / original_bytes) * 100
                    
                    f.write(f"Original Size:              {self._format_bytes(original_bytes)}\n")
                    f.write(f"Compressed Size:            {self._format_bytes(compressed_bytes)}\n")
                    f.write(f"Compression Ratio:          {ratio:.2f}x\n")
                    f.write(f"Space Saved:                {space_saved:.2f}%\n")
                f.write("\n")
            
            # Per-Signal Statistics
            if 'signals' in stats:
                f.write("PER-SIGNAL STATISTICS\n")
                f.write("=" * 80 + "\n")
                
                for signal_id, signal_stats in stats['signals'].items():
                    f.write(f"\nSignal: {signal_id}\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"  Compression Method:       {signal_stats.get('compression_method', 'N/A')}\n")
                    f.write(f"  Analytical Task:          {signal_stats.get('analytical_task', 'N/A')}\n")
                    f.write(f"  Total Ingested:           {signal_stats.get('total_ingested', 0):,} points\n")
                    f.write(f"  Total Compressed:         {signal_stats.get('total_compressed', 0):,} points\n")
                    f.write(f"  Uncompressed Buffer Size: {signal_stats.get('uncompressed_buffer_size', 0)} points\n")
                    f.write(f"  Compressed Buffer Size:   {signal_stats.get('compressed_buffer_size', 0)} segments\n")
                    
                    if 'compression_ratio' in signal_stats:
                        f.write(f"  Compression Ratio:        {signal_stats['compression_ratio']:.2f}x\n")
                    if 'space_saved_percent' in signal_stats:
                        f.write(f"  Space Saved:              {signal_stats['space_saved_percent']:.2f}%\n")
                
                f.write("\n")
            
            # Compression Methods Comparison
            if compression_results:
                f.write("COMPRESSION METHODS COMPARISON\n")
                f.write("=" * 80 + "\n")
                f.write(f"{'Method':<20} {'Ratio':<12} {'Space Saved':<15} {'Size':<20}\n")
                f.write("-" * 80 + "\n")
                
                for method, data in compression_results.items():
                    if 'error' not in data:
                        ratio = data.get('compression_ratio', 0)
                        saved = data.get('space_saved_percent', 0)
                        size = data.get('compressed_size', 0)
                        
                        f.write(f"{method.upper():<20} {ratio:<12.2f}x {saved:<14.2f}% {self._format_bytes(size):<20}\n")
                
                f.write("\n")
            
            # System Configuration
            f.write("SYSTEM CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            if 'config' in stats:
                config = stats['config']
                f.write(f"Segment Size:               {config.get('segment_size', 'N/A')}\n")
                f.write(f"Buffer Capacity:            {config.get('buffer_capacity', 'N/A')}\n")
                f.write(f"Compression Threads:        {config.get('compression_threads', 'N/A')}\n")
            f.write("\n")
            
            # Footer
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"✅ Summary report saved to: {save_path}")
    
    def generate_all_reports(
        self,
        stats: Dict[str, Any],
        compression_results: Dict[str, Dict[str, Any]] = None,
        original_size: int = None,
        compressed_size: int = None
    ):
        """
        Generate all reports at once
        
        Args:
            stats: Database statistics
            compression_results: Compression comparison results
            original_size: Original data size in bytes
            compressed_size: Compressed data size in bytes
        """
        print("\n" + "=" * 60)
        print("Generating VergeDB Performance Reports")
        print("=" * 60 + "\n")
        
        # Generate compression comparison if data available
        if compression_results:
            self.generate_compression_comparison(compression_results)
        
        # Generate storage efficiency if sizes available
        if original_size and compressed_size:
            self.generate_storage_efficiency(original_size, compressed_size)
        elif 'total_original_bytes' in stats and 'total_compressed_bytes' in stats:
            self.generate_storage_efficiency(
                stats['total_original_bytes'],
                stats['total_compressed_bytes']
            )
        
        # Generate text summary
        self.generate_summary_report(stats, compression_results)
        
        print("\n" + "=" * 60)
        print("✅ All reports generated successfully!")
        print(f"📂 Reports saved to: {self.output_dir.absolute()}")
        print("=" * 60 + "\n")
    
    def _format_bytes(self, size_bytes: int) -> str:
        """Format bytes into human-readable string"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"


# Demo function to show usage
def demo_reporter():
    """Demonstrate the reporter functionality"""
    
    # Example compression results
    compression_results = {
        'gzip': {
            'compressed_size': 12500,
            'compression_ratio': 8.0,
            'space_saved_percent': 87.5,
            'compression_time': 0.05
        },
        'gorilla': {
            'compressed_size': 15000,
            'compression_ratio': 6.67,
            'space_saved_percent': 85.0,
            'compression_time': 0.03
        },
        'sprintz': {
            'compressed_size': 14000,
            'compression_ratio': 7.14,
            'space_saved_percent': 86.0,
            'compression_time': 0.04
        },
        'paa': {
            'compressed_size': 10000,
            'compression_ratio': 10.0,
            'space_saved_percent': 90.0,
            'compression_time': 0.02
        },
        'fourier': {
            'compressed_size': 9500,
            'compression_ratio': 10.53,
            'space_saved_percent': 90.5,
            'compression_time': 0.06
        },
        'splitdouble': {
            'compressed_size': 16000,
            'compression_ratio': 6.25,
            'space_saved_percent': 84.0,
            'compression_time': 0.03
        },
        'block_subsample': {
            'compressed_size': 11000,
            'compression_ratio': 9.09,
            'space_saved_percent': 89.0,
            'compression_time': 0.02
        }
    }
    
    # Example database statistics
    stats = {
        'total_signals': 4,
        'total_data_points_ingested': 8000,
        'total_segments_compressed': 15,
        'total_segments_flushed': 10,
        'ingestion_rate_per_sec': 4523.45,
        'total_original_bytes': 100000,
        'total_compressed_bytes': 12000,
        'signals': {
            'temperature_01': {
                'compression_method': 'paa',
                'analytical_task': 'aggregation',
                'total_ingested': 2000,
                'total_compressed': 1950,
                'uncompressed_buffer_size': 50,
                'compressed_buffer_size': 3,
                'compression_ratio': 8.5,
                'space_saved_percent': 88.2
            },
            'humidity_01': {
                'compression_method': 'fourier',
                'analytical_task': 'anomaly_detection',
                'total_ingested': 2000,
                'total_compressed': 2000,
                'uncompressed_buffer_size': 0,
                'compressed_buffer_size': 4,
                'compression_ratio': 10.2,
                'space_saved_percent': 90.2
            },
            'pressure_01': {
                'compression_method': 'splitdouble',
                'analytical_task': 'aggregation',
                'total_ingested': 2000,
                'total_compressed': 2000,
                'uncompressed_buffer_size': 0,
                'compressed_buffer_size': 4,
                'compression_ratio': 6.5,
                'space_saved_percent': 84.6
            },
            'cpu_usage': {
                'compression_method': 'block_subsample',
                'analytical_task': 'classification',
                'total_ingested': 2000,
                'total_compressed': 1980,
                'uncompressed_buffer_size': 20,
                'compressed_buffer_size': 3,
                'compression_ratio': 9.1,
                'space_saved_percent': 89.0
            }
        },
        'config': {
            'segment_size': 512,
            'buffer_capacity': 50,
            'compression_threads': 3
        }
    }
    
    # Create reporter and generate all reports
    reporter = VergeDBReporter()
    reporter.generate_all_reports(stats, compression_results)


if __name__ == "__main__":
    print("VergeDB Reporter - Demonstration\n")
    demo_reporter()
