# SignSense ISL — Viva Questions and Answers

1. **What is the project objective?** Recognize selected ISL hand signs from webcam input using image processing and machine learning.
2. **Why is this an IVP project?** It uses image acquisition, color conversion, frame processing, landmark extraction and visual output.
3. **What is OpenCV used for?** Webcam acquisition, frame conversion, flipping and drawing prediction overlays.
4. **Why BGR to RGB conversion?** OpenCV uses BGR by default, while MediaPipe expects RGB input.
5. **What does MediaPipe Hands return?** 21 hand landmarks for a detected hand.
6. **How many static features are used?** 21 points × 3 coordinates = 63 features.
7. **Why normalize landmarks?** To reduce sensitivity to translation and approximate hand scale.
8. **What is supervised learning?** Learning a mapping from labeled examples to target classes.
9. **Why Random Forest?** It is a practical ensemble classifier for nonlinear feature patterns.
10. **What is a dynamic gesture?** A sign whose meaning depends partly on movement over time.
11. **How does the dynamic model represent motion?** It uses a fixed sequence of 30 landmark frames flattened into one feature vector.
12. **Why use separate static and dynamic models?** A single pose and a time sequence have different input structures.
13. **What is a train/test split?** A partition used to fit the model and estimate performance on held-out examples.
14. **Define accuracy.** Correct predictions divided by total predictions.
15. **Define precision.** Of examples predicted as a class, the fraction that are correct.
16. **Define recall.** Of true examples of a class, the fraction correctly found.
17. **What is F1-score?** The harmonic mean of precision and recall.
18. **What is a confusion matrix?** A table comparing true labels with predicted labels.
19. **Why can reported accuracy be misleading?** Similar adjacent frames can leak into train and test sets; independent sessions/signers provide stronger evaluation.
20. **Does the ZIP contain a complete ISL model?** No. It contains the training pipeline, not a broad pretrained ISL model or a large verified dataset.
21. **Can it translate full sentences?** No. Full translation needs temporal, linguistic and contextual modeling.
22. **What are key limitations?** One-hand features, user-collected vocabulary, sensitivity to lighting/occlusion and lack of facial/body features.
23. **Why log predictions?** For analysis, audit, debugging and report preparation.
24. **What is overfitting?** Good training performance but poor performance on unseen examples.
25. **What are future enhancements?** Licensed ISL datasets, two-hand/body/face features, temporal neural networks, signer-independent tests and speech output.
