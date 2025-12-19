# Diabetes Prediction using Machine Learning

A machine learning project that predicts the likelihood of diabetes in patients using Logistic Regression based on health metrics.

## Project Overview

This project implements a classification model to predict whether a patient has diabetes based on key health indicators such as Glucose levels, BMI (Body Mass Index), and Age. The model is trained on a diabetes dataset and evaluated using standard classification metrics.

## Dataset

- **File**: `diabetes.csv`
- **Features Used**: 
  - Glucose: Blood glucose level
  - BMI: Body Mass Index
  - Age: Patient's age
- **Target**: Outcome (0 = No diabetes, 1 = Diabetes)

## Project Structure

```
Diabetes_Prediction/
├── Diabetes_Prediction.ipynb    # Main Jupyter notebook with analysis and model
├── diabetes.csv                  # Dataset
└── README.md                      # Project documentation
```

## Workflow

1. **Load Dataset**: Import diabetes data from CSV file
2. **Data Exploration**: Analyze dataset structure, summary statistics, and distribution
3. **Feature Selection**: Select relevant features (Glucose, BMI, Age) for the model
4. **Train-Test Split**: Split data into 80% training and 20% testing sets
5. **Model Training**: Train Logistic Regression classifier on training data
6. **Prediction**: Generate predictions on test set
7. **Evaluation**: Assess model performance using:
   - Accuracy Score
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-Score)

## Model Details

- **Algorithm**: Logistic Regression
- **Train-Test Split**: 80-20 split with random state 42
- **Features**: 3 (Glucose, BMI, Age)
- **Output**: Binary classification (Diabetes / No Diabetes)

## Getting Started

### Requirements
- Python 3.x
- pandas
- scikit-learn

### Installation
```bash
pip install pandas scikit-learn
```

### Running the Project
1. Open `Diabetes_Prediction.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells sequentially to:
   - Load and explore the data
   - Train the model
   - Generate predictions
   - View evaluation metrics

## Results

The model outputs:
- **Accuracy Score**: Percentage of correct predictions
- **Confusion Matrix**: Breakdown of true positives, true negatives, false positives, and false negatives
- **Classification Report**: Detailed precision, recall, and F1-score for each class

## Future Enhancements

- Experiment with additional features from the dataset
- Try different classification algorithms (Decision Trees, Random Forest, SVM, etc.)
- Implement cross-validation for better model evaluation
- Add hyperparameter tuning
- Create visualizations (confusion matrix heatmap, ROC curve, etc.)
- Deploy model as a web application

## Author

Created as a Machine Learning project for portfolio/resume

## License

Open source - feel free to use and modify
