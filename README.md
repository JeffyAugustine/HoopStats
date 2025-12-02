# 🏀 HoopStats: Basketball Analysis & Tracking System

## 📋 Project Overview

**HoopStats** is a comprehensive basketball analytics platform that automates player tracking, movement analysis, and game statistics generation from video footage. The system combines computer vision, deep learning, and geometric transformations to provide detailed insights into player performance, team strategies, and game dynamics.

---

## 🎯 Key Features

- **Multi-Object Tracking**: Players & ball detection using Faster R-CNN with Deep SORT
- **Court Transformation**: Camera view to bird's-eye view perspective correction
- **Zone Analysis**: 4-court zone tracking with occupancy metrics
- **Advanced Analytics**: Speed, acceleration, possession, turnovers, passes, and player spread
- **Visualization**: Heatmaps, speed graphs, zone distributions, and annotated video output
- **Automated Statistics**: Player roles, hustle index, team performance metrics

---

## 📁 Project Structure

```
hoopstats/
│
├── 📦 Core Modules
│   ├── cav/                           # Core Analysis & Vision
│   │   ├── detection.py              # TensorFlow object detector
│   │   ├── objects.py                # BoundingBox, Object, ObjectType classes
│   │   ├── zones.py                  # Zone tracking & analytics
│   │   ├── parameters.py             # Perspective transform management
│   │   ├── visualization.py          # Map rendering & plotting
│   │   └── functions.py              # Geometric utilities
│   │
│   ├── courtvisionlib/               # Vision utilities
│   │   ├── functions.py              # Image display & frame extraction
│   │   └── helper.py                 # Deep SORT helpers & video generation
│   │
│   └── deep_sort/                    # Multi-object tracking
│       ├── detection.py              # Detection wrapper
│       ├── tracker.py                # Multi-target tracker
│       ├── track.py                  # Track management
│       ├── kalman_filter.py          # Kalman filtering
│       ├── nn_matching.py            # Nearest neighbor matching
│       ├── iou_matching.py           # IOU-based matching
│       └── linear_assignment.py      # Hungarian algorithm
│
├── 📓 Main Pipeline (Notebooks)
│   ├── 1-CameraToSky.ipynb          # Perspective transform setup
│   ├── 2-CreateDetections.ipynb     # Object detection processing
│   ├── 3-DetectZones.ipynb          # Tracking & zone analysis
│   ├── 4-GenerateVideo.ipynb        # Output video generation
│   └── analysis/                    # Advanced analytics
│       ├── Analysis.ipynb           # Visualization & insights
│       ├── Analysis-2.ipynb         # Numerical analysis & metrics
│       ├── player_speed_metrics_filtered.csv
│       ├── player_spread_analysis.csv
│       └── player_zone_metrics_cumulative.csv
│
├── 🗃️ Data & Assets
│   ├── HoopStats_assets/
│   │   ├── models/                  # Pretrained models
│   │   │   ├── frcnn/              # Faster R-CNN
│   │   │   └── mars/               # Deep SORT re-ID
│   │   └── data/                   # Intermediate results
│   │       ├── frames_raw/         # Extracted frames
│   │       ├── detections.p        # Pickled detections
│   │       ├── videopath.p         # Video path reference
│   │       └── zones_detections.csv
│   │
│   ├── data/                        # Raw video files
│   ├── icons/                       # Visualization icons
│   ├── images/                      # Reference images
│   └── tracking_output/             # Final outputs
│       ├── frames/                  # Processed frames
│       └── tracking_video.avi       # Annotated video
│
├── ⚙️ Configuration
│   ├── Q1_side_30-60.mp4           # Sample video (3840×2160)
│   ├── project_config.py           # Project settings
│   ├── params.json                 # Transform parameters
│   ├── icons_simple.json           # Icon mapping
│   └── requirements.txt            # Dependencies
│
└── 📄 README.md                    # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Installation

1. **Clone and setup environment:**
```bash
git clone https://github.com/JeffyAugustine/HoopStats.git
cd HoopStats
pip install -r requirements.txt
```

### Usage Pipeline

Run notebooks **in order**:

#### **Step 1: Camera Setup & Transformation**
```bash
jupyter notebook 1-CameraToSky.ipynb
```
- Extracts reference frames from video
- Defines 4-point perspective transform
- Creates `params.json` with transformation matrices
- Validates zone mask (5 zones: 0=background, 1-4=court zones)

![frame_view1](https://github.com/user-attachments/assets/f8cb64a4-c781-4674-9930-82ea9cc71850)

![Sky View](https://github.com/user-attachments/assets/9a10e2d6-620f-4e85-b077-6758485ad60e)

<img width="3840" height="2160" alt="mask_org" src="https://github.com/user-attachments/assets/506601ea-bf5e-49fd-a0f9-50d783973efb" />

*Above images are sample images of camera_view, Sky_view and zone_mask.*

#### **Step 2: Object Detection**
```bash
jupyter notebook 2-CreateDetections.ipynb
```
- Loads Faster R-CNN model
- Processes 721 frames at ~13.35 fps
- Detects objects (players=class 1, ball=class 37)
- Saves detections as `HoopStats_assets/data/detections.p`

#### **Step 3: Tracking & Zone Analysis**
```bash
jupyter notebook 3-DetectZones.ipynb
```
- Initializes Deep SORT tracker with MARS re-ID model
- Implements advanced features:
  - Team detection via jersey color sampling
  - Ball possession detection
  - Referee identification
  - Zone occupancy tracking
- Exports analytics to CSV

#### **Step 4: Video Generation**
```bash
jupyter notebook 4-GenerateVideo.ipynb
```
- Creates annotated video with:
  - Player bounding boxes (green)
  - Ball tracking (red)
  - Player trails (white fading lines)
  - ID labels and frame counters
- Output: `tracking_output/tracking_video.avi`

<img width="1918" height="1072" alt="image" src="https://github.com/user-attachments/assets/00ff3354-fcdf-439e-a708-c54adb7bf162" />

*Sample frame of generated video*


#### **Step 5: Analytics & Insights**
```bash
jupyter notebook analysis/Analysis.ipynb      # Visualizations
jupyter notebook analysis/Analysis-2.ipynb    # Numerical analysis
```
- Comprehensive analytics generation
- Player and team performance metrics
- Visualization plots and heatmaps

---


## 📊 Analysis Results (Sample Video: Q1_side_30-60.mp4)

### Player Performance Metrics
- **Total Players Tracked**: 13 (10 active + 3 refree)
- **Total Frames Processed**: 721 (29.44 seconds)
- **Processing Speed**: 25.40 fps (tracking phase)

### Speed Analysis (Filtered)
| Metric | Value |
|--------|-------|
| Overall Max Speed | 10.84 m/s |
| Average Player Speed | 1.94 m/s |
| Average Distance Covered | 53.24 m |

**Top 5 Fastest Players:**
1. Player 19 (Team 2): 10.84 m/s
2. Player 17 (Team 2): 10.63 m/s  
3. Player 16 (Team 2): 10.16 m/s
4. Player 15 (Team 1): 7.93 m/s
5. Player 14 (Team 2): 7.38 m/s

### Zone Occupancy
**Zone Time Distribution (All Players):**
- Right 2pt: 131.7s (44.7%)
- Right 3pt: 73.9s (25.1%)
- Left 2pt: 45.5s (15.4%)
- Left 3pt: 40.7s (13.8%)
- Outside: 2.1s (0.7%)

**Average Zone Transitions**: 7.5 per player

### Team Performance
| Metric | Team 1 | Team 2 |
|--------|--------|--------|
| **Players** | 5 | 5 |
| **Possession Time** | 22.31s (75.8%) | 7.13s (24.2%) |
| **Turnovers** | 0 | 0 |
| **Passes** | 5 successful, 2 interceptions | 1 after score, 1 after turnover |
| **Average Spread** | 5.65m | 5.38m |

### Player Roles & Hustle Index
**Top 5 Hustle Players:**
1. Player 25 (Ball): 1.87
2. Player 17 (Team 2): 0.61
3. Player 16 (Team 2): 0.60
4. Player 19 (Team 2): 0.49
5. Player 18 (Team 1): 0.48

**Player Role Classification:**
- Perimeter/3pt wing: Players 12, 15
- Inside scorer/high-post: Players 13, 14, 16, 17, 18, 19, 20, 21
- Refree/outside: Player 8, 28, 32

### Game Events
- **Score Detected**: Frame 50
- **Scoring Team**: Team 2 (Player 16)
- **Basket Location**: Box 1
- **Key Pass**: Player 17 → Player 16 (successful, led to score)

<img width="2380" height="2380" alt="Untitled design" src="https://github.com/user-attachments/assets/06203d67-d38b-4191-a6b5-0a7a6173b917" />

*Sample Visualizations*

---

## 🛠️ Technical Details

### Object Detection
- **Model**: Faster R-CNN (TensorFlow)
- **Classes**: 1 (Players), 37 (Ball)
- **Confidence Threshold**: 0.5
- **Input Resolution**: 3840×2160

### Tracking Algorithm
- **Tracker**: Deep SORT with Kalman filtering
- **Re-identification**: MARS model
- **Features**: 128-dimensional embeddings
- **Matching**: Cascade of appearance + IOU matching
- **Track States**: Tentative → Confirmed 

### Coordinate Systems
1. **Camera Coordinates**: Raw pixel positions (3840×2160)
2. **Bird's-eye Coordinates**: Transformed court view (612×433)
3. **Real-world Conversion**: 0.0645 meters per pixel
   - Court dimensions: 28.65m × 15.24m

### Zone Definitions
- **Zone 0**: Outside court
- **Zone 1**: Right 3-point area
- **Zone 2**: Right 2-point area  
- **Zone 3**: Left 3-point area
- **Zone 4**: Left 2-point area

### Advanced Features
1. **Team Detection**: Jersey color brightness analysis
2. **Ball Possession**: Player closest to ball (<50 pixels)
3. **Referee Identification**: Players spending >70% time outside court
4. **Turnover Detection**: Player + ball both in zone 0
5. **Pass Detection**: Possession changes with context awareness

---

## 📈 Output Files

### Generated Analytics
1. `player_speed_metrics_filtered.csv` - Speed & acceleration metrics
2. `player_zone_metrics_cumulative.csv` - Zone occupancy & transitions
3. `player_spread_analysis.csv` - Player spacing analysis
4. `tracking_output/tracking_video.avi` - Annotated video output
5. `tracking_output/frames/` - Individual processed frames

### Intermediate Files
1. `HoopStats_assets/data/detections.p` - Pickled detection results
2. `HoopStats_assets/data/zones_detections.csv` - Raw tracking data
3. `images/frame_view1.jpg` - Reference frame
4. `images/Sky View.jpg` - Bird's-eye court view
5. `images/mask_org.png` - Zone mask definition

---

## 🔍 Customization Guide

### For New Videos
1. **Replace video file**: Place new MP4 in root directory
2. **Update points**: Modify cameraPoints in notebook 1
3. **Adjust zone mask**: Update `images/mask_org.png` if court layout differs
4. **Re-run pipeline**: Execute notebooks 1-4 sequentially

### Parameter Tuning
| Parameter | File | Purpose |
|-----------|------|---------|
| `detection_threshold` | `cav/detection.py` | Object detection confidence |
| `max_cosine_distance` | Notebook 3 | Appearance matching threshold |
| `max_iou_distance` | Notebook 3 | IOU matching threshold |
| `max_allowed_speed_mps` | Analysis-2 | Speed filtering threshold |
| `possession_distance` | Notebook 3 | Ball possession radius |

---

## ⚡ Performance Optimization

### GPU Acceleration
- TensorFlow automatically uses available GPUs
- Set CUDA_VISIBLE_DEVICES for multi-GPU systems
- Batch size: 32 for feature extraction

### Memory Management
- Frame-by-frame processing for long videos
- Pickle files for intermediate storage
- Clear GPU memory between notebooks

### Speed vs Accuracy Trade-offs
1. **Detection phase**: ~13.35 fps (GPU dependent)
2. **Tracking phase**: ~25.40 fps
3. **Video resolution**: 3840×2160 (4K) provides best accuracy
4. **Downscaling**: Possible for faster processing

---

## 📄 License

This project is for academic and research purposes.

