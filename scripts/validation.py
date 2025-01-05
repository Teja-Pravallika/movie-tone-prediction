from impots import *

def validate(model, validation_loader, default_threshold):
    model.eval()
    total_loss = 0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for data in validation_loader:
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float32)

            logits = model(ids, mask, token_type_ids)
            loss = loss_function(logits, targets)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs >= default_threshold).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    validation_loss = total_loss / len(validation_loader)
    validation_accuracy = (np.array(all_preds) == np.array(all_targets)).mean()
    precision = precision_score(all_targets, all_preds, average='samples', zero_division=1)
    recall = recall_score(all_targets, all_preds, average='samples', zero_division=1)
    f1 = f1_score(all_targets, all_preds, average='samples', zero_division=1)

    return validation_loss, validation_accuracy, precision, recall, f1
