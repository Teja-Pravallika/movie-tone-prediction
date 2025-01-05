from imports import *

def plot_training_results(steps, train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(steps, train_losses, label='Training Loss')
    plt.plot(steps, val_losses, label='Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(steps, train_accuracies, label='Training Accuracy')
    plt.plot(steps, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
  

def find_best_thresholds(all_targets, all_probs):
    best_thresholds = {}
    for tone_idx in range(all_targets.shape[1]):
        y_true = all_targets[:, tone_idx]
        y_probs = all_probs[:, tone_idx]
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
        f1_scores = np.nan_to_num(f1_scores)  # Handle NaN values
        best_idx = np.argmax(f1_scores)
        best_thresholds[tone_idx] = thresholds[best_idx]
    return best_thresholds


def generate_pr_curves(all_targets, all_probs):
    pr_curves = {}
    for tone_idx in range(all_targets.shape[1]):
        y_true = all_targets[:, tone_idx]
        y_probs = all_probs[:, tone_idx]

        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
        auc_pr = auc(recall, precision)

        pr_curves[tone_idx] = {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'auc': auc_pr
        }
    return pr_curves

def plot_pr_curve(pr_curves, best_thresholds, tone_idx, tone_name=None):
    pr_data = pr_curves[tone_idx]
    precision, recall, thresholds = pr_data['precision'], pr_data['recall'], pr_data['thresholds']
    auc_pr = pr_data['auc']
    best_threshold = best_thresholds[tone_idx]
    best_idx = np.where(thresholds == best_threshold)[0][0]

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AUC = {auc_pr:.4f}')
    plt.scatter(recall[best_idx], precision[best_idx], color='red', label='Best Threshold')
    plt.title(f'PR Curve for {f"Tone {tone_idx}"}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid()
    plt.show()
