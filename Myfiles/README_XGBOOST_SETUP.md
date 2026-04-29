# XGBoost Event Detection Training Setup

## 🎯 Overview

This package contains everything you need to train the XGBoost event detection model from `EventDetection_Pipeline.ipynb` using your existing rugby game data.

## 📁 Folder Structure

```
.
├── Train/                          # Training data (70% of your game)
│   ├── game1.csv                   # Player tracking with velocities
│   └── RugbyEvents1.csv            # Ground-truth event labels
│
├── Evaluate/                       # Evaluation data (30% of your game)
│   ├── game2.csv                   # Player tracking with velocities
│   └── RugbyEvents2.csv            # Ground-truth event labels
│
├── prepare_xgboost_data.py         # Data preparation script
└── README_XGBOOST_SETUP.md         # This file
```

## 📊 Generated Files

### game.csv Format
```csv
frame,x,y,dx,dy,team
80,72.29,55.83,0.0,0.0,1
130,72.73,54.61,0.44,-1.22,1
...
```

**Columns:**
- `frame`: Frame number
- `x`, `y`: Player position in meters (pitch coordinates)
- `dx`, `dy`: Velocity components (meters per frame)
- `team`: Team assignment (1 or 2)

### RugbyEvents.csv Format
```csv
Event,Frame_Start,Frame_End,Team
KickRestart,60,150,1
Scrum,320,410,2
Try,500,590,1
...
```

**Columns:**
- `Event`: Event type (Scrum, Lineout, Turnover, KickRestart, Ruck, Try, OpenPlay, Halftime)
- `Frame_Start`: First frame of event
- `Frame_End`: Last frame of event
- `Team`: Team involved (1, 2, or 0 for no specific team)

## 🚀 How to Use

### Option 1: Ready-to-Use Training Data (Current Setup)

Your training data is already prepared! Just:

1. **Open the notebook:**
   ```bash
   jupyter notebook EventDetection_Pipeline.ipynb
   ```

2. **The notebook will automatically find:**
   - `Train/game1.csv` and `Train/RugbyEvents1.csv`
   - `Evaluate/game2.csv` and `Evaluate/RugbyEvents2.csv`

3. **Run all cells** to train the XGBoost model

### Option 2: Process More Games

If you have additional game footage:

```bash
python prepare_xgboost_data.py \
    --input-xy your_game_xy.csv \
    --input-events your_game_events.csv \
    --output-prefix game3 \
    --fps 30.0
```

This will create `game3.csv` and `RugbyEvents3.csv`.

### Option 3: Split New Games into Train/Eval

```bash
python prepare_xgboost_data.py \
    --input-xy your_full_game_xy.csv \
    --input-events your_full_game_events.csv \
    --split \
    --train-ratio 0.7 \
    --fps 30.0
```

This automatically creates:
- `Train/game1.csv` and `Train/RugbyEvents1.csv` (70%)
- `Evaluate/game2.csv` and `Evaluate/RugbyEvents2.csv` (30%)

## 📝 Current Training Data Stats

### Train Set (game1.csv)
- **Frames:** 30 → 530 (501 frames)
- **Player detections:** 303
  - Team 1: 201 detections
  - Team 2: 102 detections
- **Events:** 3 KickRestart events

### Eval Set (game2.csv)
- **Frames:** 60 → 230 (171 frames)
- **Player detections:** 125
  - Team 1: 88 detections
  - Team 2: 37 detections
- **Events:** 1 KickRestart event

## ⚠️ Important Notes

### Limited Event Diversity
Your current clip contains **only KickRestart events**. The XGBoost model works best with diverse training examples.

**Recommended:** Process 2-3 full-length games to get:
- Scrums
- Lineouts
- Turnovers
- Rucks
- Tries
- OpenPlay (background)

### Adding More Training Data

The notebook supports multiple games:
- `Train/game1.csv` + `Train/RugbyEvents1.csv`
- `Train/game2.csv` + `Train/RugbyEvents2.csv`
- `Train/game3.csv` + `Train/RugbyEvents3.csv`
- etc.

Just run `prepare_xgboost_data.py` on each game with `--output-prefix game2`, `--output-prefix game3`, etc.

## 🔧 Data Preparation Script Details

The `prepare_xgboost_data.py` script:

1. **Computes velocities (dx, dy)**
   - Per-track frame-to-frame deltas
   - Outlier filtering (>20m/frame jumps)

2. **Maps event types**
   - Your pipeline → Notebook format
   - `Stop` → `OpenPlay` (background class)

3. **Generates synthetic Try events**
   - Based on ball position near tryline
   - Ball speed < 0.5 m/s

4. **Adds OpenPlay samples**
   - Every 5 seconds where no event
   - Creates background/negative class

## 🎓 Next Steps

1. **Run the notebook** with current data to test the pipeline
2. **Process full games** to get diverse event types
3. **Review Phase 1-5 plan** in your gap analysis document
4. **Implement Phase 1** (dx/dy in production pipeline)

## 🐛 Troubleshooting

### "No events found in RugbyEvents.csv"
- Make sure your `*_events.csv` file contains detected events
- The script maps your event types to notebook format

### "Not enough training samples"
- Process more game footage
- Each game should be 10+ minutes for good coverage

### "Team assignment incorrect"
- Script maps `L` → Team 1, `R` → Team 2
- Check your team codes in `*_events.csv`

## 📧 Questions?

Review the gap analysis document for the roadmap from rule-based to ML-based event detection.

---

**Generated:** 2026-04-16  
**Source Data:** cliped_raw_20260412_133511_765122  
**Multi-Sport Platform:** Basketball | Soccer | RugbyUnion | RugbyLeague
