# Human-Action-Video-Classification

UCF101 Video Classification and Style Transfer

This project demonstrates a two-stage deep learning pipeline:
	1.	Video Classification: Using ResNet with transfer learning on the UCF101 dataset.
	2.	Style Transfer: Applying a CycleGAN model to stylize videos while preserving their semantic content.
 
🧠 Stage 1: Video Classification (ResNet)

I classify human actions using transfer learning on ResNet:
	•	Dataset: UCF101 – 101 human action categories in videos.
	•	Preprocessing: Sampled video frames resized to feed into ResNet.
	•	Model: Pre-trained ResNet50 with a custom classifier head.
	•	Training: Fine-tuned on a subset of UCF101.
 
📊 Model Evaluation
Accuracy: 96%

🎨 Stage 2: Style Transfer (CycleGAN)

After classification, I stylize videos with CycleGAN:
	•	Objective: Transfer video appearance to another visual domain (e.g., from natural to artistic style) without paired data.
	•	Architecture: Cycle-consistent adversarial networks with ResNet-based generators.
	•	Result: Maintain content (pose/action) while transforming style.

✨ Features
	•	Frame-by-frame transformation using pre-trained CycleGAN weights.
	•	Option to run in real-time via streamlit_cyclegan_video.py.
