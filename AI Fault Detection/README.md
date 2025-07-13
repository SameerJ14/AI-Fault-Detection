
# AI-Powered Fault Detection in Electronic Circuits

This project uses a Random Forest classifier to detect faults in an electronic circuit based on simulated sensor readings (voltage, current, temperature).

## 📁 Files
- `circuit_faults.csv`: Simulated dataset (500 entries)
- `model_training.py`: Python script to train and evaluate the model
- `confusion_matrix.png`: Confusion matrix from evaluation

## 🔧 Tools & Libraries
- Python
- Pandas, Scikit-learn, Matplotlib, Seaborn

## 🚀 How to Run
1. Install required libraries:
```bash
pip install pandas scikit-learn matplotlib seaborn
```
2. Run the training script:
```bash
python model_training.py
```

## 📈 Results
Achieved ~92% accuracy in classifying faulty vs. non-faulty states. Confusion matrix visualizes model performance.

## 📌 Author
Sameer Jawale — Master's in Electrical Engineering, UM-Dearborn
