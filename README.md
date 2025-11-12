# XAI for WPW

This repository contains code for the **Explainable Deep Learning-based Classification of Wolff-Parkinson-White Electrocardiographic Signals** paper (https://arxiv.org/pdf/2511.05973). The focus is on classifying **12-lead ECG signals**, each consisting of **200 time steps**, while employing **explainable AI (XAI)** to uncover the link between the features driving the decision-making process of the deep learning (DL) models and physiological events in the electrical activation of the heart. All models were implemented using the **TensorFlow** deep learning framework.

---

## Overview
 
- **Goal:** Detect WPW patterns and gain insights into the ECG features that are influential to the decision-making process of the DL model.  
- **Framework:** TensorFlow / Keras  
- **Explainability:** Grad-CAM (cam.py), Guided Grad-CAM (cam.py), Guided Backpropagation (guidedbackprop.py) 
- **Output:** Model weights (run.py, tune_stack.py, tune_multichannel.py, tune_image.py), predictions (run.py), saliency maps (cam.py, guidedback.py)  

---

## Requirements

After creating a virtual environment, install dependencies running:

```bash
pip install -r requirements.txt
```
---

## Usage
To tune the 2D FCN model and also output the selected hyperparameters, from the scripts folder run:
```bash
python3 tune_image.py --run True
```
Running
```bash
python3 tune_image.py --run False
```
will print the selected hyperparameters after the tuning has been completed once. The same holds for tune_stack.py and tune_multichannel.py.

To train and finetune a model (for instance, the 2D FCN) run:
```bash
python3 run.py --model image
```
Other possible values for the model argument are 'stack' and 'multichannel'.

Finally, guidedbackprop.py and cam.py extract, respectively, the saliency maps for the Guided Backpropagation method and for the Grad-CAM method. Combining them via the element-wise product will yield the Guided Grad-CAM maps. In this work, all maps were later analyzed considering their absolute value.

