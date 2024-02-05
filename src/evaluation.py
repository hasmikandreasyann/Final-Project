# src/evaluation.py

from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    classification_rep = classification_report(true_labels, predictions)
    
    return accuracy, classification_rep

if __name__ == "__main__":
    # Example usage
    example_predictions = [1, 0, 1, 1, 0]
    example_true_labels = [1, 0, 1, 1, 1]

    # Example of model evaluation
    accuracy, classification_report = evaluate_model(example_predictions, example_true_labels)

    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report)
