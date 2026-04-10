import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, classification_report

def calculate_metrics(csv_path):
    print(f"Reading data from {csv_path}...")
    try:
        # Read the CSV
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find the file {csv_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file {csv_path} is empty.")
        return

    # Check if necessary columns exist
    if 'true_label' not in df.columns or 'predicted_label' not in df.columns:
        print("Error: The CSV file must contain 'true_label' and 'predicted_label' columns.")
        return

    # Extract true and predicted labels
    y_true = df['true_label'].astype(str)
    y_pred = df['predicted_label'].astype(str)

    # Get unique labels (classes)
    labels = sorted(list(set(y_true) | set(y_pred)))

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Calculate F1 score (macro and weighted)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Calculate classification report which gives F1 per class too
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    print(f"\n--- Metrics for {csv_path} ---")
    print("\nConfusion Matrix:")
    
    # Print a nicely formatted confusion matrix using pandas
    cm_df = pd.DataFrame(cm, index=[f"True {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
    print(cm_df.to_string())

    print(f"\nF1 Score (Macro):    {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate confusion matrix and F1 score from a CSV file.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file (e.g., tests.csv)")
    args = parser.parse_args()

    calculate_metrics(args.csv_path)
