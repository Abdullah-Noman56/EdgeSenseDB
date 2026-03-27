# EdgeSenseDB / VergeDB

A lightweight, adaptive time-series database designed specifically for Edge and IoT environments constrained by CPU, memory, and storage limits. This project implements dynamic compression techniques, an adaptive resource controller, and a custom query engine to optimize data storage and retrieval on resource-limited devices.

## 🚀 Features

*   **Adaptive Resource Controller:** Dynamically monitors system resources (CPU, Memory) and intelligently switches compression algorithms.
*   **Advanced Data Compression:** Custom algorithms tailored for time-series sensor data such as temperature, pressure, and humidity to lower storage footprints.
*   **Custom Query Engine:** Capable of executing efficient data retrieval directly from compressed or uncompressed states without massive overhead.
*   **Comprehensive Reporting:** Includes automated demo and reporting scripts for performance analysis.
*   **IEEE Publication Draft:** Companion IEEE-formatted paper detailing the methodology and performance gains achieved via our adaptive database architecture.

## 📁 Repository Structure

```
├── code/
│   ├── adaptive_controller.py      # Core resource monitoring and adaptation logic
│   ├── advanced_compression.py     # Custom sensor data compression algorithms
│   ├── query_engine.py             # Custom engine for executing data retrieval
│   ├── verge_database.py           # Database operations and storage handling
│   ├── vergedb_demo.py             # Main entry point for demonstrations
│   ├── demo_with_reports.py        # Demo script generating reports
│   └── vergedb_reporter.py         # Module to export system performance metrics
├── Paper/
│   └── i232078_i232110_i232012_IEEE_latex.txt   # Draft of the IEEE architecture paper
└── README.md
```

## 🛠️ Usage

To run the demonstration and see the adaptive compression in action:

1.  Make sure you have Python 3.8+ installed.
2.  Navigate to the `code/` directory.
3.  Run the main demo script:
    ```bash
    python vergedb_demo.py
    ```
4.  Optionally, run the full demo with metrics reporting:
    ```bash
    python demo_with_reports.py
    ```

## 📄 IEEE Paper

This repository contains the IEEE formal manuscript detailing the system architecture, mathematical formulations for the adaptive controller, and empirical results. You can find the raw LaTeX file under `Paper/`.

## 👥 Authors
* Student Habib Ahmed
* Student M.Akash Waris
* Student Abdullah Noman

## 📝 License

This project is released for academic and research purposes.
