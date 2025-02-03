import time
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from model import BNext4DFR  # Import the model class from model.py


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_dataloader(data_dir, batch_size=32, shuffle=False):
    dataset = datasets.ImageFolder(root=data_dir, transform=get_transforms())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)


def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred, y_logits = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            logits = outputs["logits"].squeeze()
            preds = (torch.sigmoid(logits) > 0.5).int()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_logits.extend(torch.sigmoid(logits).cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_logits)
    f1 = f1_score(y_true, y_pred)
    return acc, auc, f1


def main():
    dataset_dir = "D:/Facultate/Licenta/binary_deepfake_detection/DatasetTrim"
    train_dir, val_dir, test_dir = [os.path.join(dataset_dir, x) for x in ["Train", "Validation", "Test"]]

    batch_size = 64
    num_classes = 2
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Initialize the model
    model = BNext4DFR(num_classes=num_classes, add_magnitude_channel=True, add_fft_channel=True, add_lbp_channel=True).to(device)

    # Load the checkpoint
    checkpoint_path = "D:/Facultate/Licenta/binary_deepfake_detection/pretrained/tiny_checkpoint.pth.tar"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint, strict=False)  # Allow partial loading
        print("Checkpoint loaded successfully.")
    else:
        print(f"Warning: Checkpoint file '{checkpoint_path}' not found. Running with untrained weights.")

    # Create data loaders
    train_loader = get_dataloader(train_dir, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_dir, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_dir, batch_size=batch_size, shuffle=False)

    start = time.time()
    # Evaluate the model
    print("Evaluating on validation set...")
    val_acc, val_auc, val_f1 = evaluate_model(model, val_loader, device)
    print(f"Validation - ACC: {val_acc:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}")

    print("Evaluating on test set...")
    test_acc, test_auc, test_f1 = evaluate_model(model, test_loader, device)
    print(f"Test - ACC: {test_acc:.4f}, AUC: {test_auc:.4f}, F1: {test_f1:.4f}")
    finish = time.time()

    duration = finish - start
    print(duration)


if __name__ == "__main__":
    main()