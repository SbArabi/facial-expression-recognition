# Facial Expression Recognition

Real-time 7-emotion classifier using **MediaPipe face detection** → **MobileNet CNN** → **Softmax Regression Calibrator**. Runs entirely on CPU at ~5 FPS.

Detects: **angry · disgust · fear · happy · neutral · sad · surprise**

---

## How It Works — End to End

Every frame from your webcam goes through this pipeline:

```ascii
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Webcam    │────▶│  Face        │────▶│  Emotion          │────▶│  Calibrator      │
│   640×480   │     │  Detection   │     │  Classifier       │     │  (Softmax Reg.)  │
│             │     │  (MediaPipe) │     │  (TFLite MobileNet)│     │  7×7 weight      │
└─────────────┘     └──────────────┘     └──────────────────┘     └──────────────────┘
                                                                          │
                                                                          ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  On-screen  │◀────│  Dashboard   │◀────│  Emotion         │
│  Display    │     │  Renderer    │     │  Tracker         │
│  (OpenCV)   │     │  (bars,chart,│     │  (3-frame avg    │
│             │     │   stats)     │     │   + 30s history) │
└─────────────┘     └──────────────┘     └──────────────────┘
```

### Step 1 — Face Detection (MediaPipe)

MediaPipe's **BlazeFace** model (single-shot detector) scans the camera frame for faces. It outputs a **bounding box** — the (x, y, width, height) of the largest face in view. This runs in ~5ms on a CPU.

```
┌─────────────────────────┐
│                         │     The model was trained on 6K+
│    ┌──────────┐         │     face images at multiple scales.
│    │  FACE    │         │     It detects faces at any angle,
│    │  BBOX    │         │     with glasses, partial occlusion.
│    │          │         │
│    └──────────┘         │
│                         │
└─────────────────────────┘
        640×480
```

### Step 2 — Face Preprocessing

The detected face region is:
1. **Cropped** from the full frame
2. **Resized** to **224×224 pixels** (the input size the MobileNet expects)
3. **Converted** from BGR to RGB color space
4. Passed to the model **as raw pixel values (0–255)** — no normalization is applied

```python
# Simplified version of what happens:
face_crop = frame[y1:y2, x1:x2]          # Crop to bounding box
rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)  # BGR → RGB
input_tensor = cv2.resize(rgb, (224, 224))         # Scale to 224×224
```

### Step 3 — CNN Emotion Classification (MobileNet)

The 224×224 RGB face is fed into a **MobileNet** convolutional neural network fine-tuned on **AffectNet** (400,000+ real-world face images). Here's what happens inside:

```
┌──── 224×224×3 RGB face ────┐
            │
     ┌──────▼──────┐
     │  Conv Layer 1│──▶ Edge detection (horizontal/vertical lines, 
     │  (3×3×32)    │    corners, color blobs)
     └──────────────┘
            │
     ┌──────▼──────┐
     │  Conv Layer 2│──▶ Texture detection (eye shapes, mouth curves,
     │  (3×3×64)    │    wrinkle patterns, eyebrow angles)
     └──────────────┘
            │
     ┌──────▼──────┐
     │  Conv Layer 3│──▶ Part detection (left eye position, right eye,
     │  (3×3×128)   │    nose bridge, lips corner, chin line)
     └──────────────┘
            │
          ··· (more convolutional layers downsample to 7×7×1024)
            │
     ┌──────▼──────┐
     │  Global Avg  │──▶ Collapse spatial dimensions → 1024 features
     │  Pooling     │
     └──────────────┘
            │
     ┌──────▼──────┐
     │  Dense Layer │──▶ Each of the 7 emotions gets a raw score (logit)
     │  (7 neurons) │
     └──────────────┘
            │
     ┌──────▼──────┐
     │  Softmax    │──▶ Converts scores to probabilities (sum = 1.0)
     └──────────────┘
            │
     ┌──────▼──────┐
     │  [0.02,      │  ← angry=2%, disgust=1%, fear=35%,
     │   0.01,      │     happy=3%, neutral=5%, sad=10%,
     │   0.35, ...] │     surprise=44%
     └──────────────┘
```

**What each layer learns:**

| Stage | Output Size | What It Detects |
|-------|-----------|-----------------|
| Initial conv layers | 112×112 | Edges, gradients, color transitions |
| Mid conv layers | 28×28 | Eye shapes, mouth curvature, brow angles |
| Deep conv layers | 7×7 | High-level combinations: "wide-open eyes + raised brows + open mouth = fear or surprise" |
| Global pooling | 1024 | Single feature vector summarizing the entire face |
| Dense (logits) | 7 | Raw score per emotion |
| Softmax | 7 | Probability per emotion (sum to 100%) |

**Why convolutional layers work:** Each filter slides across the image looking for a specific pattern. Early filters find simple patterns (horizontal lines, edges). Deeper filters combine those into complex patterns (an eye, a mouth shape). The final layer maps these patterns to emotion scores.

**The depthwise separable convolution trick:** MobileNet uses a special "depthwise separable" convolution that splits the operation into two lighter steps — this is what makes it fast enough to run on a CPU in real-time.

### Step 4 — Why Fear, Surprise, and Sad Get Confused

After the softmax, the model outputs 7 probabilities. For most people, the top emotion gets ~50–95% confidence. But **fear, surprise, and sad** are often confused because:

```
Feature      Fear      Surprise     Sad
───────────  ────────  ──────────  ────────
Eyebrows     Raised↑   Raised↑↑    Inner brows up↑
Eyes         Wide      Very wide   Slightly droopy↓
Mouth        Stretched  Open wide   Corners down↓
Forehead     Wrinkled  Smooth      Relaxed
```

The model's deep layers activate similarly for fear and surprise because **both have raised eyebrows, wide eyes, and open mouths**. The difference is subtle — surprise has higher eyebrows and a more rounded mouth — but the 48×48 low-resolution training images don't capture these details well.

Our test data showed:
- **44.7%** of surprise images were predicted as fear
- **24%** of fear images were predicted as sad
- The model defaults to "fear" when uncertain (fear bias)

### Step 5 — Softmax Regression Calibrator (The Fix)

A simple per-class scale+bias can't fix confusion *between* emotions — it only amplifies or suppresses each emotion independently. We need something smarter.

**Softmax regression** learns a **7×7 weight matrix** where every weight represents "how does emotion A's activation influence emotion B's final score?":

```ascii
                Corrected scores
              ┌──────────────────┐
              │  angry   │  W₁₁  │  ← influenced by raw angry score
              │  disgust │  W₂₂  │  ← influenced by raw disgust score
              │  fear    │  W₃₃  │  ← influenced by raw fear score
Input (7 raw  │  happy   │= W₄₄  │  ← influenced by raw happy score
probabilities)│  neutral │  W₅₅  │  ← influenced by raw neutral score
              │  sad     │  W₆₆  │  ← influenced by raw sad score
              │  surprise│  W₇₇  │  ← influenced by raw surprise score
              └──────────────────┘
                        │
            PLUS cross terms (the key insight):
              ┌──────────────────┐
              │  angry   │+ W₁₃  │  ← also influenced by fear score
              │  disgust │+ W₂₇  │  ← also influenced by surprise score
              │  fear    │+ W₃₇  │  ← also influenced by surprise score
              │  surprise│+ W₇₃  │  ← also influenced by fear score
              └──────────────────┘
```

**The critical correction:** The weight `W_surprise_from_fear` learns that "when fear probability is high AND surprise probability is also high → it's probably surprise, not fear." This is something a simple scale+bias cannot learn.

**Training data:**
- **560 FER2013 images** (80 per emotion) — gray, 48×48, low quality, but lots of them
- **Your Pinterest images** augmented to 60 samples — high-quality real faces, specifically for the confusing emotions

**Training process:** Mini-batch gradient descent with L2 regularization, 500 epochs, minimizing cross-entropy loss.

**Before vs After calibrated confusion:**

```ascii
BEFORE (raw model):               AFTER (calibrated):
   True\Pred  sad fear surp         True\Pred  sad fear surp
        sad   48%   3%  11%              sad   63%  14%   2%
       fear   17%  33%  26%             fear   24%  33%  22%
    surp      0%  45%  41%          surp      2%  14%  79%
```

### Step 6 — Temporal Smoothing

The calibrated probabilities go through a **3-frame rolling average**. This prevents the displayed emotion from flickering between frames when the model is near a decision boundary. The chart and stats use the smoothed values. The face overlay label uses the **raw (unsmoothed) argmax** for instant responsiveness.

---

## Performance

### Per-Class Accuracy

| Emotion   | Raw Model | Calibrated | Improvement |
|-----------|-----------|------------|------------|
| Surprise  | 40.6%     | **79.2%**  | **+38.5%** |
| Sad       | 50.9%     | **63.0%**  | **+12.0%** |
| Angry     | 43.8%     | **48.8%**  | +5.0%      |
| Happy     | 80.0%     | **81.2%**  | +1.2%      |
| Disgust   | 67.5%     | **67.5%**  | —          |
| Neutral   | 52.5%     | **53.8%**  | +1.2%      |
| Fear      | 44.8%     | 33.3%      | -11.5%     |
| **Overall** | **53.5%** | **60.8%**  | **+7.3%** |

### Confusion Matrix (Calibrated)

| True\Pred | angry | disgust | fear | happy | neutral | sad | surprise |
|-----------|-------|---------|------|-------|---------|-----|----------|
| angry     | 48.8% | 11.2%   | 13.8%| 1.2%  | 11.2%   | 7.5%| 6.2%     |
| disgust   | 10.0% | 67.5%   | 6.2% | 3.8%  | 3.8%    | 7.5%| 1.2%     |
| fear      | 9.4%  | 3.1%    | 33.3%| 3.1%  | 5.2%    | 24.0%| 21.9%   |
| happy     | 0.0%  | 0.0%    | 10.0%| 81.2% | 2.5%    | 0.0%| 6.2%     |
| neutral   | 6.2%  | 1.2%    | 1.2% | 7.5%  | 53.8%   | 26.2%| 3.8%    |
| sad       | 4.6%  | 1.9%    | 13.9%| 0.9%  | 13.9%   | 63.0%| 1.9%     |
| surprise  | 3.1%  | 0.0%    | 13.5%| 1.0%  | 1.0%    | 2.1%| 79.2%    |

**The critical improvement:** Surprise→Fear confusion dropped from **44.7%→13.5%**. Sad→Surprise confusion dropped from **10.7%→1.9%**.

### Runtime

| Step | Time per frame |
|------|---------------|
| Face detection (MediaPipe) | ~5ms |
| Face crop + resize | ~1ms |
| TFLite inference (224×224) | ~180ms |
| Calibrator + smoothing | <0.1ms |
| Dashboard rendering | ~10ms |
| **Total** | **~200ms (~5 FPS)** |

---

## Quick Demo

```bash
# 1. Create environment
python -m venv fer_face_env

# 2. Activate it
fer_face_env\Scripts\activate

# 3. Install dependencies
pip install opencv-python mediapipe ai-edge-litert numpy

# 4. Run
fer_face_env\Scripts\python src\webcam_demo.py
```

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Toggle recording (saves video + CSV with emotion timeline) |
| `s` | Toggle side-by-side mode (raw camera vs dashboard) |

---

## Project Structure

```
facial-expression-recognition/
│
├── src/                          # Source code
│   ├── webcam_demo.py            # Entry point — imports and runs main()
│   ├── main.py                   # Orchestrator: connects all modules
│   ├── config.py                 # Constants: labels, colors, thresholds, layout
│   ├── face_detector.py          # MediaPipe face detection wrapper
│   ├── emotion_classifier.py     # TFLite model inference wrapper
│   ├── emotion_tracker.py        # 3-frame smoothing + 30s history buffer
│   ├── stats_tracker.py          # Session statistics (dominant, transitions)
│   ├── ui_dashboard.py           # OpenCV renderer: camera, bars, chart, stats
│   └── recorder.py               # Video + CSV recording with codec fallback
│
├── models/                       # Pre-trained models
│   ├── emotiefflib_mobilenet_7.tflite   # MobileNet CNN (AffectNet, 224×224)
│   ├── calibrator.pkl                    # Softmax regression weights
│   └── face_detection_short_range.tflite # MediaPipe BlazeFace model
│
├── .gitignore
└── README.md
```

---

## Technical Details

### Model Architecture

| Property | Value |
|----------|-------|
| Base model | MobileNet (depthwise separable convolutions) |
| Pre-training | VGGFace2 (face identification, 3.3M images) |
| Fine-tuning | AffectNet (400K images, 7 emotions) |
| Input size | 224×224×3 RGB |
| Input range | [0, 255] raw pixels (no normalization) |
| Output | 7-class softmax probabilities |
| Format | TensorFlow Lite (integer-only quantized) |
| Benchmark | 64.71% on AffectNet 7-class test set |

### Why MobileNet?

MobileNet uses **depthwise separable convolutions** — a standard 3×3 conv is split into:
1. A depthwise conv (one filter per input channel)
2. A 1×1 pointwise conv (combines channels)

This reduces computation by 8–9× compared to a standard convolution with minimal accuracy loss, making real-time CPU inference possible.

### From Face to Emotion — The Full Chain

```
Original frame (640×480)
    │
    ▼
MediaPipe detects face → returns bounding box [x, y, w, h]
    │
    ▼
Face region cropped from frame
    │
    ▼
Cropped face resized to 224×224, converted to RGB
    │
    ▼
TFLite MobileNet processes 224×224×3 tensor
    ├── Conv layers extract features (edges → textures → face parts)
    ├── Global pooling collapses to 1024-d feature vector
    └── Dense layer + softmax → 7 probabilities
    │
    ▼
Softmax regression calibrator adjusts probabilities
    ├── Standardizes (z-score using training stats)
    ├── Linear transform (7×7 weight matrix + bias)
    └── Softmax → calibrated probabilities
    │
    ▼
EmotionTracker receives calibrated probabilities
    ├── Rolling 3-frame average (for chart + stats)
    └── Raw argmax stored separately (for face overlay)
    │
    ▼
StatsTracker updates session stats
    ├── Dominant emotion
    ├── Emotion transitions
    └── Time distribution
    │
    ▼
UIDashboard renders the display
    ├── Camera feed with emotion label overlay
    ├── Probability bars (7 horizontal bars)
    ├── Emotion history chart (30s scrollable)
    └── Session statistics panel
```

### The Calibrator — Mathematical Formulation

Given raw model output vector **x** ∈ ℝ⁷:

1. **Standardize:** `x_norm = (x - μ) / σ` (μ, σ learned from training)
2. **Augment:** `x_aug = [x_norm, 1.0]` (adds bias term)
3. **Linear:** `logits = x_aug · W` (W is 8×7 weight matrix)
4. **Softmax:** `y_calibrated = softmax(logits)`

The weight matrix **W** has 56 parameters (7×8). Each entry `W[i][j]` represents how much class i's raw score influences class j's final probability. The off-diagonal entries learn cross-class corrections — for example, a positive entry from "fear" to "surprise" means "when fear is also high, increase surprise's score."

### Training Setup

The calibrator was trained on a CPU using pure NumPy (no TensorFlow required for inference):

| Hyperparameter | Value |
|---------------|-------|
| Training samples | 620 (560 FER2013 + 60 curated) |
| Model | Softmax regression (multinomial logistic regression) |
| Loss | Cross-entropy + L2 regularization (λ=0.001) |
| Optimizer | Mini-batch SGD (batch size=128, lr=0.1) |
| Epochs | 500 (early stopping with lr decay) |
| Implementation | Pure NumPy (no GPU needed) |

---

## Requirements

| Library | Version | Purpose |
|---------|---------|---------|
| opencv-python | ≥4.5 | Camera capture, image processing, display |
| mediapipe | ≥0.10 | Face detection |
| ai-edge-litert | — | TFLite model inference |
| numpy | — | Array operations, calibrator math |

---

## License

This project uses the **EmotiEffLib MobileNet** model from the [EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib) research project by Samara University (sb-ai-lab). The model is pre-trained on VGGFace2 and fine-tuned on AffectNet. The calibration code and training scripts are original work.
