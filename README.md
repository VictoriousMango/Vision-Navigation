# 🚀 Vision-Navigation: Modular Visual SLAM Suite

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![VSLAM](https://img.shields.io/badge/Visual_SLAM-Simulation_and_Analysis-green)
![Streamlit](https://img.shields.io/badge/Interactive-Dashboard-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A powerful, modular Python-based Visual SLAM (Simultaneous Localization and Mapping) framework for **feature detection**, **pose estimation**, **loop closure**, and **3D trajectory visualization**. Features a Streamlit interface, support for ROS2 integration, and full KITTI dataset compatibility.

---

## 🧭 Key Features

- 🔍 Feature Detectors: SIFT, ORB, BRISK, Affine-SIFT/ORB
- 🧩 Modular Pipeline (`Pipeline` class) for VSLAM tasks
- 🎯 Loop Detection with Bag-of-Visual-Words (BoVW)
- 📈 Pose Estimation and Error Analytics
- 🧪 Batch Testing of KITTI Sequences
- 📊 Streamlit Visual Dashboard (`pages/`)
- 🐳 Docker and ROS2 Launch Support

---

## 🗂️ Repository Structure

```bash
Vision-Navigation/
├── assets/                # Core VSLAM Modules (Frontend, Mapping, Loop Closure)
├── pages/                 # Streamlit Visual Dashboard Pages
├── Archieve/             # Intermediate JSON datasets
├── results/              # CSV files of trajectory outputs and MSE results
├── logs/                 # Logs & HTML summaries
├── ROS2/                 # Docker & ROS2 integration configs
├── Literature/           # Experimental notebooks and theory modules
├── VisualBoG.py          # BoVW Vocabulary Creator
├── DemoVSLAM.py          # End-to-end demonstration script
├── Dockerfile            # For containerized deployment
├── STARTER.bat           # Windows startup utility
├── config.py             # Configuration and hyperparameters
├── requirements.txt      # All Python dependencies
└── README.md             # You're here!
```

---

## 🧠 VSLAM Workflow

```mermaid
flowchart LR
    A[Input RGB Images] --> B[Feature Detection]
    B --> C[Feature Matching]
    C --> D[Pose Estimation (E/F Matrix)]
    D --> E[Triangulation]
    E --> F[Local Mapping]
    F --> G[Loop Detection (BoVW)]
    G --> H[Trajectory Optimization]
    H --> I[3D Map + Trajectory Output]
```

---

## ⚙️ Usage Example

```python
from assets.FrontEnd_Module import Pipeline
pipe = Pipeline()
trajectory = pipe.VisualOdometry(frame, FeatureDetector='FD_SIFT')
```

For BoVW vocabulary building:

```python
from VisualBoG import generateCodeBook
generateCodeBook(numOfWords=500)
```

---

## 🧪 Streamlit Dashboard (`pages/`)

| Page                   | Description                                           |
|------------------------|-------------------------------------------------------|
| `Automated_tests.py`   | Batch VO testing + error plots                        |
| `Simulation.py`        | End-to-end sequence + loop detection                  |
| `Test_SingleImage.py`  | Inspect feature & descriptor matching                 |
| `viewTestFiles.py`     | Visualize and compare pose CSVs                       |

Run the Streamlit app:

```bash
streamlit run pages/Simulation.py
```

---

## 🐳 Docker + ROS2 Support

Build Docker image:

```bash
docker build -t vslam-docker .
```

Launch ROS2:

```bash
cd ROS2
source Docker2Xlaunch.md
```

---

## 📚 References

- KITTI Odometry Dataset
- OpenCV Feature Detection
- scikit-learn KMeans
- Streamlit Docs

---

## ✅ Pros & ❌ Cons

| ✅ Pros                                     | ❌ Cons                                       |
|--------------------------------------------|-----------------------------------------------|
| Modular & easy to extend                   | No depth sensor → scale ambiguity             |
| Multiple FD/FM options with visual output  | Performance depends on calibration            |
| Integrated Streamlit UI                    | Repetitive textures may reduce accuracy       |
| BoVW-based loop detection                  | Hardware intensive for large sequences        |

---