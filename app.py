import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd

st.set_page_config(page_title="ROC AUC Curve Demo", layout="wide")

st.title("ROC AUC Curve Demonstration")
st.markdown("""
This interactive app demonstrates how the **Receiver Operating Characteristic (ROC)** curve
and **Area Under the Curve (AUC)** work for binary classification problems.
""")

# Sidebar controls
st.sidebar.header("Configuration")

# Choose scenario
scenario = st.sidebar.selectbox(
    "Select Scenario",
    ["Custom Data", "Perfect Classifier", "Random Classifier", "Good Classifier", "Poor Classifier"]
)

# Number of samples
n_samples = st.sidebar.slider("Number of Samples", 10, 1000, 500, 10)

# Prevalence rate (proportion of positive samples)
prevalence = st.sidebar.slider("Prevalence Rate (% Positive)", 0.1, 0.9, 0.5, 0.05)

# Calculate number of positive and negative samples
n_positive = int(n_samples * prevalence)
n_negative = n_samples - n_positive

# Generate data based on scenario
np.random.seed(42)

if scenario == "Custom Data":
    st.sidebar.subheader("Custom Parameters")
    mean_positive = st.sidebar.slider("Mean (Positive Class)", 0.0, 1.0, 0.7, 0.05)
    mean_negative = st.sidebar.slider("Mean (Negative Class)", 0.0, 1.0, 0.3, 0.05)
    std_positive = st.sidebar.slider("Std Dev (Positive)", 0.01, 0.5, 0.2, 0.01)
    std_negative = st.sidebar.slider("Std Dev (Negative)", 0.01, 0.5, 0.2, 0.01)

    # Generate scores
    y_true = np.concatenate([np.ones(n_positive), np.zeros(n_negative)])
    y_scores = np.concatenate([
        np.random.normal(mean_positive, std_positive, n_positive),
        np.random.normal(mean_negative, std_negative, n_negative)
    ])
    y_scores = np.clip(y_scores, 0, 1)

elif scenario == "Perfect Classifier":
    y_true = np.concatenate([np.ones(n_positive), np.zeros(n_negative)])
    y_scores = y_true.copy()

elif scenario == "Random Classifier":
    y_true = np.concatenate([np.ones(n_positive), np.zeros(n_negative)])
    y_scores = np.random.rand(n_samples)

elif scenario == "Good Classifier":
    y_true = np.concatenate([np.ones(n_positive), np.zeros(n_negative)])
    y_scores = np.concatenate([
        np.random.normal(0.75, 0.15, n_positive),
        np.random.normal(0.25, 0.15, n_negative)
    ])
    y_scores = np.clip(y_scores, 0, 1)

else:  # Poor Classifier
    y_true = np.concatenate([np.ones(n_positive), np.zeros(n_negative)])
    y_scores = np.concatenate([
        np.random.normal(0.55, 0.2, n_positive),
        np.random.normal(0.45, 0.2, n_negative)
    ])
    y_scores = np.clip(y_scores, 0, 1)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Threshold selection (before plotting so we can mark it on the ROC curve)
st.sidebar.markdown("---")
st.sidebar.markdown("### Threshold Selection")

# Show threshold selection method
threshold_method = st.sidebar.radio(
    "Select threshold by:",
    ["Index", "Value"],
    horizontal=True
)

if threshold_method == "Index":
    threshold_idx = st.sidebar.slider(
        "Threshold Index",
        0,
        len(thresholds) - 1,
        len(thresholds) // 2,
        format=f"%d of {len(thresholds) - 1}"
    )
    selected_threshold = thresholds[threshold_idx]
else:
    # For value-based selection, use a slider with the threshold range
    # Filter out infinite values (sklearn's roc_curve can return inf)
    finite_mask = np.isfinite(thresholds)
    finite_thresholds = thresholds[finite_mask]

    if len(finite_thresholds) > 0:
        min_threshold = float(finite_thresholds.min())
        max_threshold = float(finite_thresholds.max())
    else:
        # Fallback if all thresholds are infinite (unlikely)
        min_threshold = 0.0
        max_threshold = 1.0

    selected_threshold = st.sidebar.slider(
        "Threshold Value",
        min_threshold,
        max_threshold,
        (min_threshold + max_threshold) / 2,
        0.01
    )
    # Find the closest threshold index (search in all thresholds, including inf)
    threshold_idx = np.argmin(np.abs(thresholds - selected_threshold))
    selected_threshold = thresholds[threshold_idx]  # Use the actual threshold from the curve

selected_fpr = fpr[threshold_idx]
selected_tpr = tpr[threshold_idx]

# Display both index and value for reference
threshold_display = f"{selected_threshold:.4f}" if np.isfinite(selected_threshold) else f"{selected_threshold}"
st.sidebar.info(f"**Index:** {threshold_idx} / {len(thresholds) - 1}\n\n**Threshold:** {threshold_display}")

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("ROC Curve")

    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5)')

    # Plot the selected threshold point
    ax.scatter(selected_fpr, selected_tpr, color='red', s=100, zorder=5,
               label=f'Selected Threshold', marker='o', edgecolors='darkred', linewidths=2)

    # Add annotation for the threshold point
    threshold_text = f"{selected_threshold:.3f}" if np.isfinite(selected_threshold) else f"{selected_threshold}"
    ax.annotate(f'Threshold = {threshold_text}\nTPR = {selected_tpr:.3f}\nFPR = {selected_fpr:.3f}',
                xy=(selected_fpr, selected_tpr),
                xytext=(20, -20),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red'),
                fontsize=9)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

with col2:
    st.subheader("Metrics")
    st.metric("AUC Score", f"{roc_auc:.4f}")

    # Display dataset info
    st.markdown("---")
    st.markdown("### Dataset Info")
    st.write(f"**Prevalence:** {prevalence:.1%}")
    st.write(f"**Positive Samples:** {n_positive}")
    st.write(f"**Negative Samples:** {n_negative}")

    # Display threshold metrics
    st.markdown("---")
    st.markdown("### Threshold Metrics")
    threshold_display_metric = f"{selected_threshold:.4f}" if np.isfinite(selected_threshold) else f"{selected_threshold}"
    st.write(f"**Threshold:** {threshold_display_metric}")
    st.write(f"**TPR (Sensitivity):** {selected_tpr:.4f}")
    st.write(f"**FPR:** {selected_fpr:.4f}")
    st.write(f"**Specificity:** {1 - selected_fpr:.4f}")

# Score distribution plot
st.subheader("Score Distribution by True Class")

fig2, ax2 = plt.subplots(figsize=(10, 4))
positive_scores = y_scores[y_true == 1]
negative_scores = y_scores[y_true == 0]

ax2.hist(negative_scores, bins=30, alpha=0.6, label='Negative Class', color='blue', edgecolor='black')
ax2.hist(positive_scores, bins=30, alpha=0.6, label='Positive Class', color='red', edgecolor='black')
ax2.axvline(selected_threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold = {selected_threshold:.3f}')
ax2.set_xlabel('Prediction Score', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Distribution of Prediction Scores', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

st.pyplot(fig2)

# Confusion Matrix at selected threshold
st.subheader("Confusion Matrix at Selected Threshold")

y_pred = (y_scores >= selected_threshold).astype(int)
cm = confusion_matrix(y_true, y_pred)

col3, col4 = st.columns([1, 2])

with col3:
    # Display confusion matrix as table
    cm_df = pd.DataFrame(
        cm,
        index=['Actual Negative', 'Actual Positive'],
        columns=['Predicted Negative', 'Predicted Positive']
    )
    st.dataframe(cm_df)

    tn, fp, fn, tp = cm.ravel()
    st.write(f"**True Negatives:** {tn}")
    st.write(f"**False Positives:** {fp}")
    st.write(f"**False Negatives:** {fn}")
    st.write(f"**True Positives:** {tp}")

with col4:
    # Plot confusion matrix
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax3.figure.colorbar(im, ax=ax3)

    ax3.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'],
            title='Confusion Matrix',
            ylabel='True label',
            xlabel='Predicted label')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16)

    st.pyplot(fig3)

# Educational content
with st.expander("What is ROC AUC?"):
    st.markdown("""
    ### ROC Curve
    The **Receiver Operating Characteristic (ROC)** curve is a graphical plot that illustrates
    the diagnostic ability of a binary classifier as its discrimination threshold is varied.

    - **X-axis (FPR):** False Positive Rate = FP / (FP + TN)
    - **Y-axis (TPR):** True Positive Rate (Sensitivity/Recall) = TP / (TP + FN)

    ### AUC Score
    The **Area Under the Curve (AUC)** represents the degree or measure of separability.
    It tells how much the model is capable of distinguishing between classes.

    - **AUC = 1.0:** Perfect classifier
    - **AUC = 0.5:** Random classifier (no discrimination ability)
    - **AUC < 0.5:** Worse than random (predictions are inverted)
    - **0.5 < AUC < 1.0:** Better than random (higher is better)

    ### Interpretation
    - **0.9 - 1.0:** Excellent
    - **0.8 - 0.9:** Good
    - **0.7 - 0.8:** Fair
    - **0.6 - 0.7:** Poor
    - **0.5 - 0.6:** Fail

    ### Prevalence
    **Prevalence** is the proportion of positive cases in the dataset (P / (P + N)).

    - An important note: **ROC curves are insensitive to class imbalance**
    - AUC remains the same regardless of prevalence (though other metrics like precision change)
    - Prevalence affects the interpretation of threshold-specific metrics
    - Real-world applications often have very different prevalence than 50/50
    """)

with st.expander("How to Use This App"):
    st.markdown("""
    1. **Select a Scenario:** Choose from predefined scenarios or create custom data
    2. **Adjust Parameters:** Use the sidebar to modify:
       - Number of samples
       - Prevalence rate (proportion of positive cases)
       - Distribution parameters (for Custom Data)
    3. **Observe the ROC Curve:** See how the curve changes with different data
       - Note: ROC curve is insensitive to prevalence changes
    4. **Select a Threshold:** Use the slider to see how different thresholds affect predictions
       - The red point on the ROC curve shows your selected threshold
    5. **Analyze Metrics:** View TPR, FPR, confusion matrix, and dataset composition
    """)
