from scripts.model import initialize_model
from scripts.dataset import create_dataloaders, tokenizer, max_token_length
from scripts.validation import validate


if __name__ == "__main__":
    validation_loss, validation_accuracy, precision, recall, f1 = validate(model, testing_loader, default_threshold)

    print(f"Final Test Metrics:")
    print(f"Loss = {validation_loss:.4f}")
    print(f"Accuracy = {validation_accuracy * 100:.2f}%")
    print(f"Precision = {precision:.4f}")
    print(f"Recall = {recall:.4f}")
    print(f"F1 Score = {f1:.4f}")
