# MSK Multimodal Data Collection & Analysis

This repository contains tools and scripts for processing and analyzing multimodal musculoskeletal (MSK) data, including motion capture (MoCap), computer vision-based pose estimation, and joint angle calculations.

## 📋 Overview

This project enables comprehensive biomechanical analysis through multiple data collection modalities:
- **Motion Capture (MoCap)** systems for high-precision 3D movement tracking
- **Computer Vision** approaches using MediaPipe for markerless pose estimation
- **Joint Angle Analysis** for biomechanical assessment
- **Statistical Analysis** for comparing different measurement modalities

## 🏗️ Project Structure

```
msk_multimodal_data_collection/
├── Python/                              # Python-based analysis tools
│   ├── mediapipe_batch_process.py       # Batch processing of videos with MediaPipe
│   ├── functions.py                     # Core biomechanical calculation functions
│   ├── data_reading.py                  # Data loading and preprocessing utilities
│   ├── data_processing_figures.py       # Data visualization and figure generation
│   ├── create_figures_newest.py         # Advanced figure creation tools
│   ├── recursive_directory_processor.py # Automated data processing workflows
│   ├── requirements.txt                 # Python package dependencies
│   └── Joint Angle Results/             # Statistical analysis and results
│       ├── joint_angle_comparisons.py   # Cross-modal joint angle comparisons
│       ├── new_statistical_testing_*.py # Statistical validation scripts
│       ├── table_preparation_paper.py   # Publication-ready table generation
│       └── clean_statistical_results.py # Results post-processing
├── Matlab/                              # MATLAB analysis tools
│   ├── Joint_Angle.m                    # Main joint angle calculation script
│   ├── my3Dangle.m                      # 3D angle computation functions
│   ├── mybutter.m                       # Butterworth filtering implementation
│   └── MSK_Keyfile.*                    # Configuration files for data processing
└── README.md                            # This documentation
```

## 🔧 Core Functionality

### 1. Computer Vision-Based Pose Estimation
- **MediaPipe Integration**: Automated pose landmark detection from video data
- **Batch Processing**: Process multiple videos simultaneously
- **Body Part Tracking**: 33 key body landmarks including joints, extremities, and facial features
- **Export Capabilities**: Generate CSV files with 3D coordinate data

### 2. Joint Angle Calculations
- **3D Biomechanical Analysis**: Calculate joint angles in 3D space
- **Multi-Joint Support**: Analyze major joints (shoulder, elbow, hip, knee, ankle)
- **Filtering Options**: Butterworth filtering for noise reduction
- **Cross-Platform**: Available in both Python and MATLAB implementations

### 3. Data Processing & Analysis
- **Multi-Modal Comparison**: Compare MoCap vs. computer vision results
- **Statistical Validation**: Comprehensive statistical testing (t-tests, Wilcoxon, correlation analysis)
- **Movement Analysis**: Support for common exercises (squats, sit-to-stand movements)
- **Data Visualization**: Generate publication-ready figures and plots

### 4. Automated Workflows
- **Recursive Processing**: Automatically process entire directory structures
- **Standardized Naming**: Consistent data organization and column naming
- **Batch Analysis**: Process multiple participants and trials efficiently

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- MATLAB R2020a+ (for MATLAB components)
- Required Python packages (see `requirements.txt`)

### Installation

1. **Clone the repository**:
   ```bash
   git clone [REPOSITORY_URL]
   cd msk_multimodal_data_collection
   ```

2. **Install Python dependencies**:
   ```bash
   cd Python
   pip install -r requirements.txt
   ```

3. **Download MediaPipe model** (if needed):
   - Ensure `pose_landmarker_heavy.task` is in the Python directory

### Basic Usage

#### Process Videos with MediaPipe
```python
python mediapipe_batch_process.py
# Follow prompts to specify video directory and output location
```

#### Calculate Joint Angles (MATLAB)
```matlab
run('Matlab/Joint_Angle.m')
% Enter data path and configuration when prompted
```

#### Generate Comparison Analysis
```python
python Joint\ Angle\ Results/joint_angle_comparisons.py
# Specify root data folder for analysis
```

## 📊 Data Organization

### Expected Input Structure
```
data/
├── MoCap/
│   ├── Participant_01_MoCap/
│   │   ├── Squat.csv
│   │   ├── Sit To Stand.csv
│   │   └── Sqt To Box.csv
│   └── Participant_02_MoCap/
└── ComputerVision/
    ├── Participant_01_NC/
    │   ├── Squat.csv
    │   ├── Sit To Stand.csv
    │   └── Sqt To Box.csv
    └── Participant_02_NC/
```

### Supported Movement Types
- **Squat**: Deep knee bend movements
- **Sit To Stand**: Transitional movements from seated to standing
- **Sqt To Box**: Controlled squat movements to a target

## 📈 Analysis Capabilities

### Statistical Measures
- Paired t-tests for related samples
- Wilcoxon signed-rank tests for non-parametric data
- Pearson correlation analysis
- Descriptive statistics (mean, std, min, max, range)

### Visualization Options
- Joint angle time series plots
- Comparative box plots
- Correlation scatter plots
- Statistical significance tables

## 📁 Dataset Information

> **Note**: Links to datasets and contact information will be added here.

### Dataset Access
- **Primary Dataset**: [LINK TO BE ADDED]
- **Supplementary Data**: [LINK TO BE ADDED]
- **Data Documentation**: [LINK TO BE ADDED]

### Contact for Data Access
- **Principal Investigator**: [NAME TO BE ADDED]
- **Email**: [EMAIL TO BE ADDED]
- **Institution**: [INSTITUTION TO BE ADDED]
- **Data Repository**: [REPOSITORY TO BE ADDED]

### Data Usage Guidelines
- [USAGE TERMS TO BE ADDED]
- [CITATION REQUIREMENTS TO BE ADDED]
- [ETHICAL APPROVAL INFORMATION TO BE ADDED]

## 🔬 Research Applications

This toolkit has been used for:
- Validating computer vision against gold-standard motion capture
- Clinical movement assessment studies
- Biomechanical research in rehabilitation settings
- Remote movement analysis applications

## 📝 Citation

If you use this toolkit in your research, please cite:
```
[CITATION TO BE ADDED]
```

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed descriptions

## 📄 License

[LICENSE TO BE ADDED]

## 🆘 Support

For technical support or questions:
- **Issues**: Create an issue in this repository
- **Email**: [SUPPORT EMAIL TO BE ADDED]
- **Documentation**: See individual script headers for detailed usage

## 🔄 Version History

- **v1.0**: Initial release with core functionality
- **Current**: Enhanced statistical analysis and visualization tools

---

*This README provides an overview of the MSK multimodal data collection and analysis toolkit. For detailed usage instructions, refer to individual script documentation and comments.*
