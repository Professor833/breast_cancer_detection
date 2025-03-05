# Breast Cancer Detection

This project implements machine learning models to detect breast cancer based on features extracted from breast mass images.

## Project Structure

```
.
├── dataset
│   └── breast_cancer.csv
├── results
│   └── logistic_regression_results.png
├── README.md
├── logistic_regression.py
└── requirements.txt
```

## Dataset

The dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast masses. Features include characteristics like radius, texture, perimeter, area, smoothness, etc. The target variable indicates whether the mass is benign (0) or malignant (1).

## Models

### Logistic Regression

The `logistic_regression.py` script implements:
- Data preprocessing and feature scaling
- Model training with regularization
- Comprehensive evaluation metrics:
  - Accuracy
  - Confusion matrix
  - Classification report (precision, recall, F1-score)
  - Cross-validation
  - ROC curve and AUC

## Requirements

```
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
```

## Usage

```bash
# Install requirements
pip install -r requirements.txt

# Run logistic regression model
python logistic_regression.py
```

## Results

The model evaluation results, including the confusion matrix and ROC curve, are saved in the `results` directory.

## Future Work

- Implement additional models (Random Forest, SVM, Neural Networks)
- Feature selection techniques
- Hyperparameter tuning
- Model interpretability analysis
