# SignSense ISL — Advanced IVP Project
Members- Samradhi Jain, Girishma Goyal

## What this project does
- Real-time webcam hand tracking using MediaPipe Hands.
- Static sign classifier trained from your own labeled examples.
- Dynamic gesture classifier trained from 30-frame hand-landmark sequences.
- Dataset capture, model persistence, hold-out metrics, confusion matrix and prediction CSV logs.
- Tkinter desktop dashboard.

## Compatibility
Use **64-bit Python 3.12** and `mediapipe==0.10.21`. This is deliberately pinned because newer MediaPipe package versions may not expose the legacy `mp.solutions` API used here. No Docker is needed.

## Windows setup
Open PowerShell in the extracted project folder:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install --upgrade pip
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -c "import mediapipe as mp, cv2; print(mp.__version__, hasattr(mp, 'solutions'), cv2.__version__)"
.\.venv312\Scripts\python.exe main.py
```
Expected MediaPipe check: `0.10.21 True ...`

Always launch with `.\.venv312\Scripts\python.exe main.py` to avoid accidentally using the old Python 3.14 environment. If Python 3.12 is not installed, install it from https://www.python.org/downloads/release/python-31210/ and enable the Python Launcher.

## Workflow
1. Run the app and choose **Live Detection → Start Camera**.
2. Open **Dataset Studio** and enter a label. Use ISL vocabulary labels only if your captured pose/movement matches that sign.
3. For static signs, hold a pose and capture individual samples or batches of 30/80. Collect at least 40–80 examples per label initially, from varied hand positions and sessions.
4. For dynamic gestures, click **Record 2-second gesture** and perform the entire motion. Repeat 20–50 sequences per label; include at least two different labels.
5. Go to **Train & Evaluate** and train the static and/or dynamic model.
6. Test live. Export prediction logs from Live Detection.
7. Review reported accuracy, precision, recall, F1 and confusion matrix.

## Data files
- `data/static_landmarks.csv`: each row has label, 63 normalized landmark features and timestamp.
- `data/dynamic_sequences.npz`: fixed-length 30-frame sequences and labels.
- `models/static_rf.joblib`, `models/dynamic_rf.joblib`: trained models.
- `logs/predictions.csv`: timestamped predicted labels, confidence and mode.

## Important limits — read before presenting
This is a **dataset-trained prototype**, not a pre-trained complete ISL translator. The ZIP does not include a large verified ISL dataset or pretrained ISL model. It will only recognize labels you collect and train. Static landmark features do not encode facial expression or full body posture. The dynamic model uses hand landmark sequences only and may not capture two-handed signs. Full ISL sentence translation needs a legally usable, appropriately labeled ISL dataset, signer-diverse evaluation, temporal modeling, two-hand/body/face features where needed, and language/grammar modeling. Do not report an accuracy until you have collected data and trained the model.

## Troubleshooting
- `No module named mediapipe`: install dependencies using the `.venv312` interpreter shown above.
- `mp.solutions` missing: confirm MediaPipe is 0.10.21 and run the exact `.venv312` interpreter.
- Camera unavailable: close other apps using the camera; edit `cv2.VideoCapture(0, cv2.CAP_DSHOW)` to camera index 1 if needed.
- Low accuracy: add more balanced examples, separate training and test recording sessions, and avoid near-duplicate frames leaking across splits.
