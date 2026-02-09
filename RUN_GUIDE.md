# Crossy Road Body Pose Controller - Run Guide

Control Crossy Road using your body movements detected through webcam!

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Webcam** connected and working
3. **MediaPipe Pose Landmarker model** downloaded

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install opencv-python mediapipe numpy pynput
```

### 2. Download MediaPipe Model

Download the pose landmarker model and place it in the `models/` directory:

1. Create the models directory (if it doesn't exist):
   ```bash
   mkdir -p models
   ```

2. Download the model from MediaPipe:
   - Visit: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models
   - Download `pose_landmarker_heavy.task` or `pose_landmarker_full.task`
   - Rename it to `pose_landmarker.task`
   - Place it in the `models/` directory

Alternative wget command:
```bash
wget -O models/pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

### 3. Directory Structure

Your project should look like:
```
Crossy-Road-Jumps/
├── models/
│   └── pose_landmarker.task
├── src/
│   └── main.py
├── requirements.txt
└── RUN_GUIDE.md
```

## Running the Controller

### 1. Start the Program

```bash
python src/main.py
```

### 2. Calibration Phase (First 2 seconds)

- Stand in front of the camera in a neutral position
- Stay still while the system calibrates
- You'll see "CALIBRATING: X/2.0s" at the bottom of the screen
- Once calibrated, status changes to "STATUS: CALIBRATED"

### 3. Control Scheme

The screen is divided into three vertical zones:

**LEFT ZONE** | **MIDDLE ZONE** | **RIGHT ZONE**

#### How to Play:

1. **Jump**: Perform a real jump (bend knees and jump up)
   - Always presses **Up** arrow key

2. **Jump + Left**: Jump while your torso is in the LEFT zone
   - Presses **Up** + **Left** arrow keys

3. **Jump + Right**: Jump while your torso is in the RIGHT zone
   - Presses **Up** + **Right** arrow keys

4. **Jump + Forward**: Jump while in the MIDDLE zone
   - Presses only **Up** arrow key

#### Tips:
- Lean your body left or right to change zones (the system tracks your hip center)
- Jump naturally - the system detects upward velocity
- Wait for cooldown between jumps (~0.5 seconds)
- Keep your full body visible in the frame

### 4. HUD Information

The bottom overlay shows:
- **STATUS**: Calibration status
- **ZONE**: Current zone (LEFT/MIDDLE/RIGHT) with color coding
- **DISPLACEMENT**: Vertical displacement from baseline
- **VELOCITY**: Current vertical velocity (negative = moving up)

### 5. Exit

Press **'q'** key in the OpenCV window to quit

## Troubleshooting

### Camera Not Working
- Check if camera index is correct (default is 0)
- Change `CAMERA_INDEX` constant in src/main.py if needed

### No Pose Detection
- Ensure good lighting
- Stay within camera frame
- Move closer to camera
- Check if model file exists at `models/pose_landmarker.task`

### Jump Detection Too Sensitive/Not Sensitive Enough

Edit constants in `src/main.py`:

```python
# Make jumps easier to trigger (decrease threshold)
JUMP_VELOCITY_THRESHOLD = 0.010  # Default: 0.015

# Make jumps harder to trigger (increase threshold)
JUMP_VELOCITY_THRESHOLD = 0.020

# Adjust minimum displacement
MIN_JUMP_DISPLACEMENT = -0.03  # Default: -0.05 (less negative = easier)
```

### Multiple Jumps from Single Jump

Increase cooldown:
```python
JUMP_COOLDOWN_SECONDS = 0.8  # Default: 0.5
```

### Zone Detection Issues

Adjust smoothing (lower = more responsive, higher = more stable):
```python
POSITION_SMOOTHING = 0.5  # Default: 0.7
```

## Game Setup

1. Launch Crossy Road game
2. Position the game window so you can see both the game and the controller window
3. Make sure the game window is focused when you want to play
4. Start jumping and moving!

## Performance Tips

- Close other applications to free up CPU/GPU
- Use good lighting for better pose detection
- Ensure camera has clear view of your full body
- Stand 5-8 feet from camera for best results

## Configuration

All tunable parameters are at the top of `src/main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_INDEX` | 0 | Camera device index |
| `CALIBRATION_SECONDS` | 2.0 | Calibration duration |
| `JUMP_VELOCITY_THRESHOLD` | 0.015 | Upward velocity to trigger jump |
| `JUMP_COOLDOWN_SECONDS` | 0.5 | Time between jumps |
| `MIN_JUMP_DISPLACEMENT` | -0.05 | Minimum upward movement |
| `POSITION_SMOOTHING` | 0.7 | Position filtering (0-1) |
| `VELOCITY_SMOOTHING` | 0.8 | Velocity filtering (0-1) |

## Enjoy!

Have fun playing Crossy Road with your body! Jump and lean to navigate your character across the roads and rivers.
