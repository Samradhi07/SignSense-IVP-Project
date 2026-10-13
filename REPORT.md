# SignSense ISL: Real-Time Sign Language Recognition using Image and Video Processing

## Abstract
SignSense ISL is a desktop prototype for collecting and classifying selected Indian Sign Language (ISL) hand signs from a webcam. OpenCV acquires video frames; MediaPipe Hands detects 21 hand landmarks; normalized landmark vectors are classified by Random Forest models. The project includes separate static-pose and dynamic-sequence training workflows, hold-out evaluation metrics, a prediction dashboard and CSV logs.

## Problem statement
People unfamiliar with ISL can face communication barriers. A vision-based learning prototype can demonstrate recognition of a limited, well-defined vocabulary. A broad translator requires substantially more labeled data and modeling than a classroom prototype.

## Objectives
1. Acquire live frames from a webcam.
2. Extract and normalize hand landmark features.
3. Capture labeled examples for static and dynamic signs.
4. Train supervised classifiers.
5. Evaluate models using accuracy, precision, recall, F1 and confusion matrices.
6. Display predictions and store timestamped logs.
7. Document limitations and future extensions honestly.

## Tools and requirements
- Python 3.12, Windows 10/11, webcam
- OpenCV, MediaPipe 0.10.21, NumPy, Pandas, scikit-learn, Joblib, Pillow, Tkinter
- Recommended: 8 GB RAM and stable lighting

## System pipeline
```text
Webcam → OpenCV frame acquisition → BGR/RGB conversion
       → MediaPipe hand landmarks (21 points)
       → Wrist-origin and scale normalization
       → Static 63-feature vector / dynamic 30-frame sequence
       → Random Forest classifier
       → Label and confidence estimate → Dashboard + CSV log
```

## Image processing and feature extraction
OpenCV reads BGR frames and converts them to RGB for MediaPipe. For a detected hand, the application extracts 21 landmarks with x, y and z coordinates. The wrist point is subtracted from all coordinates to reduce translation effects. The points are divided by a scale estimate derived from the maximum 2D distance from the wrist. This yields 63 normalized features for static classification.

## Static classification
Each captured sample contains a class label, 63 normalized features and a timestamp. A Random Forest with 300 trees is trained using a stratified 80/20 split when possible. The application displays accuracy and a per-class precision/recall/F1 report plus a confusion matrix.

## Dynamic gesture classification
The capture workflow records approximately two seconds of landmark frames, resamples them to 30 time steps, and stores the sequence. The dynamic Random Forest consumes the flattened 30 × 63 feature array. It can distinguish temporal patterns only when those patterns are represented in the training data. This first implementation is a baseline; sequence models such as LSTM/Temporal CNN/Transformer may be explored later.

## Implemented features
- Webcam acquisition and hand landmarks
- Static and dynamic custom data capture
- Random Forest training for both feature types
- Hold-out accuracy, precision, recall, F1 and confusion matrix
- Live prediction display and confidence estimate
- Timestamped prediction logs and CSV export
- Desktop dashboard and project documentation

## Data-dependent features / not implemented as universal capability
The package does not include a pretrained, broad-coverage ISL model or a large verified ISL dataset. The user must capture/import licensed and correctly labeled data. The current feature extractor handles one hand; signs requiring two hands, facial grammar, body pose or non-manual markers will not be reliably represented. Full sentence translation, grammar/context modeling, text-to-speech and robust signer-independent recognition are future work.

## Evaluation protocol
Record the number of examples per label, label count, accuracy, per-class precision/recall/F1, and confusion matrix. For credible evaluation, collect independent sessions and ideally different signers for testing. A random split of near-identical adjacent frames can overestimate generalization. Do not fill in results before training.

### Results table (complete after running)
- Number of static labels: ______
- Static sample count: ______
- Static hold-out accuracy: ______
- Number of dynamic labels: ______
- Dynamic sequence count: ______
- Dynamic hold-out accuracy: ______
- Main confusion pairs: ______
- Test signers/sessions: ______

## Limitations
1. Recognition is restricted to labels in the user's dataset.
2. The feature representation is hand-only and single-hand.
3. Dynamic sequence baseline can be sensitive to speed and missing frames.
4. Evaluation quality depends on label balance and independent test data.
5. Confidence values from Random Forest are not guaranteed calibrated probabilities.
6. It is not a certified or complete sign-language translation system.

## Future scope
- Add an appropriately licensed ISL dataset importer and verified label mapping.
- Include both hands, pose and face features when required by signs.
- Use temporal deep learning for dynamic signs.
- Test across signers, lighting and recording sessions.
- Add phrase assembly, text-to-speech and accessible UI.
- Add dataset visualizations and reproducible experiment exports.

## Conclusion
SignSense ISL demonstrates an end-to-end IVP workflow for static hand-pose and baseline dynamic gesture classification. It connects image acquisition, landmark-based feature extraction, supervised learning, evaluation and live inference. Broader ISL coverage depends on appropriate data, sign-language expertise and temporal/contextual modeling.
