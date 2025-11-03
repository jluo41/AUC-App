# ROC AUC Curve Demonstration App

An interactive Streamlit application to demonstrate how Receiver Operating Characteristic (ROC) curves and Area Under the Curve (AUC) work for binary classification problems.

## Features

- **Multiple Scenarios**: Choose from predefined scenarios (Perfect, Random, Good, Poor classifiers) or create custom data
- **Interactive ROC Curve**: Visualize the ROC curve with real-time AUC calculation
- **Threshold Selection**: Adjust the classification threshold and see its impact on metrics
- **Score Distribution**: View how prediction scores are distributed for positive and negative classes
- **Confusion Matrix**: See the confusion matrix at any selected threshold
- **Educational Content**: Learn about ROC curves, AUC scores, and their interpretations

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

Run the Streamlit app with:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## How to Use

1. **Select a Scenario** from the sidebar (or choose "Custom Data" to create your own)
2. **Adjust Parameters** like number of samples and distribution characteristics
3. **Observe the ROC Curve** and how AUC changes with different data
4. **Select a Threshold** using the slider to see performance at that operating point
5. **Analyze Metrics** including TPR, FPR, Specificity, and Confusion Matrix

## Understanding ROC AUC

- **ROC Curve**: Shows the trade-off between True Positive Rate and False Positive Rate
- **AUC Score**: Measures the ability to distinguish between classes (0.5 = random, 1.0 = perfect)
- **Threshold**: The decision boundary for converting prediction scores to class labels

## Requirements

- Python 3.8+
- Streamlit
- NumPy
- Matplotlib
- Scikit-learn
- Pandas
