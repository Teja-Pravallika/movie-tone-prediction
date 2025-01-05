from impots import *
from scripts.model import create_dataloaders, initialize_model, tokenizer, max_token_length
from scripts.validation import validate

def train_model(model, training_loader, validation_loader, optimizer, num_epochs, default_threshold):
    steps = []
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()

        total_loss, total_accuracy, steps_completed = 0, 0, 0

        for step, data in enumerate(training_loader, 1):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float32)

            logits = model(ids, mask, token_type_ids)
            loss = loss_function(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            probs = torch.sigmoid(logits.detach().cpu())
            preds = (probs >= default_threshold).float()
            targets_cpu = targets.detach().cpu()
            batch_accuracy = (preds == targets_cpu).sum().item() / targets.numel()
            total_accuracy += batch_accuracy
            steps_completed += 1

            if step % 100 == 0:
                train_loss = total_loss / steps_completed
                train_acc = total_accuracy / steps_completed

                val_loss, val_acc, _, _, _ = validate(model, validation_loader, default_threshold)

                steps.append((epoch * len(training_loader)) + step)
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)

                print(f"Step {step}: Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        epoch_train_loss = total_loss / steps_completed
        epoch_train_acc = total_accuracy / steps_completed
        val_loss, val_acc, precision, recall, f1 = validate(model, validation_loader, default_threshold)
        print(f"Epoch {epoch + 1} Training Loss: {epoch_train_loss:.4f}, Training Accuracy: {epoch_train_acc:.4f}")
        print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        print(f"Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    
    print("Training complete.")
    return steps, train_losses, val_losses, train_accuracies, val_accuracies

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classification")
    args = parser.parse_args()

    steps, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, training_loader, testing_loader, optimizer, args.num_epochs, args.threshold
    )
