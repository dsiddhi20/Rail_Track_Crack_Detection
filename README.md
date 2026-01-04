# 🚆 Rail Track Crack Detection using Machine Learning

## 📌 Project Overview
Railway track cracks can lead to serious accidents if not detected early.  
This project detects cracks on railway tracks using **image processing and machine learning**, helping improve railway safety through early fault identification.

---

## 🎯 Objective
To classify railway track images into:
- Defective (Crack present)
- Non Defective (No crack)

using machine learning techniques.

---

## 🧰 Technologies Used
- Python  
- OpenCV  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Streamlit  

---

## 📂 Dataset
- **Source:** Railway Track Fault Detection Dataset (Kaggle)  
- **Link:** https://www.kaggle.com/datasets/salmaneunus/railway-track-fault-detection  

The dataset is pre-organized into:
- Train
- Validation
- Test  

Each set contains:
- Defective images  
- Non Defective images  

⚠️ Dataset files are excluded from this repository using `.gitignore` due to size constraints.

---

## 🧪 Methodology
1. Loaded railway track images from dataset  
2. Converted images to grayscale and resized them  
3. Extracted pixel-level features  
4. Trained a **Support Vector Machine (SVM)** classifier  
5. Evaluated performance using validation and test sets  
6. Deployed the trained model using **Streamlit**  

---

## 📊 Results
- The model successfully detects cracks in railway track images  
- Achieved reliable accuracy on unseen test data  
- Visual results help in understanding model predictions  

---

## 💻 Streamlit Application
The project includes a Streamlit web application that allows users to:
- Upload a railway track image  
- Get real-time crack detection results  

To run the app:
```bash
streamlit run app.py
