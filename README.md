# FACETRACK: Intelligent Real-Time Attendance System

A real-time face recognition attendance system built with Machine Learning.

## Accuracy
- KNN + PCA: 100%
- SVM + PCA: 100%
- Random Forest: 100%

## Tech Stack
Python | OpenCV | PCA | KNN | SVM | Random Forest | Tkinter | CSV | Excel

## Features
- Real-time multi-face detection using Haar Cascade
- PCA dimensionality reduction retaining 95% variance
- Three ML models trained and compared
- Auto attendance marking with name, roll number, department
- Saves attendance to both CSV and Excel automatically
- Clean dark themed Tkinter desktop UI

## How to Run
pip install opencv-python scikit-learn pillow openpyxl
python app.py

## Project Structure
- capture_faces.py - Capture face photos via webcam
- train_model.py - Train PCA + ML models
- recognize.py - Real time recognition
- app.py - Full Tkinter desktop application

## Research
Research paper presented at University level and submitted
to Scopus-indexed conference proceedings.

## Author
Jyoti | B.Sc Computer Science, University of Delhi
GitHub: github.com/jyoti-analytics
