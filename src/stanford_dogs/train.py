import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from stanford_dogs.model import build_model
from stanford_dogs.dataset import get_dataloaders
from stanford_dogs.transforms import get_transforms
from stanford_dogs.config import EPOCHS, LR, CHECKPOINTS_DIR, LOGS_DIR, DATA_DIR, NUM_WORKERS, BATCH_SIZE, SEED, NUM_CLASSES, IMG_SIZE
from stanford_dogs.utils import seed_everything, configure_logging

from tqdm import tqdm
import time
import logging

logger = logging.getLogger(__name__)

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    loss_value = 0.0
    num_correct = 0
    len_data = len(train_loader.dataset)

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device, non_blocking = True), labels.to(device, non_blocking = True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        loss_value += loss.item()*inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        num_correct += (preds == labels).sum().item()

    epoch_loss = loss_value/len_data
    epoch_acc = num_correct/len_data

    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    loss_value = 0.0
    num_correct = 0
    len_data = len(val_loader.dataset)

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device, non_blocking = True), labels.to(device, non_blocking = True)
    
            outputs = model(inputs)
            loss = criterion(outputs, labels)
    
            loss_value += loss.item()*inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            num_correct += (preds == labels).sum().item()
    
        val_loss = loss_value/len_data
        val_acc = num_correct/len_data
    
    return val_loss, val_acc

def main(args: argparse.Namespace | None = None):
    if args is None:
        args = parse_args()

    seed_everything(seed= SEED)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info("Training on device: %s", device)
        
    train_transform, test_transform = get_transforms(img_size= IMG_SIZE)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_transform= train_transform,
        test_transform= test_transform,
        batch_size= args.batch_size,
        num_workers= args.num_workers,
        seed= SEED,
        data_dir= DATA_DIR,
    )

    model = build_model(num_classes= NUM_CLASSES).to(device)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = args.learning_rate,
    )
    criterion = nn.CrossEntropyLoss()

    epochs = args.n_epochs
    best_val_loss = float("inf")
    CHECKPOINTS_DIR.mkdir(parents= True, exist_ok= True)
    best_model_path = CHECKPOINTS_DIR / "best_model.pt"

    logger.info("Starting training for %d epochs...", epochs)
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model= model, train_loader= train_loader, optimizer=optimizer, criterion=criterion, device=device,
        )
        val_loss, val_acc = validate(
            model=model, val_loader=val_loader, criterion=criterion, device=device,
        )
        epoch_duration = time.time() - epoch_start

        logger.info(
            "Epoch [%02d/%02d] (%.1fs) | Train Loss: %.4f - Train Acc: %.2f%% | Val Loss: %.4f - Val Acc: %.2f%%",
            epoch + 1, epochs, epoch_duration, train_loss, train_acc * 100, val_loss, val_acc * 100
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                best_model_path,
            )
            logger.info("  --> Saved new best checkpoint to %s",best_model_path)

    total_time = time.time() - start_time
    logger.info("Training complete in %.2f minutes.", total_time/60)

    # 7. Final test set evaluation
    logger.info("Evaluating best model on unseen Test Set...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = validate(model, test_loader, criterion, device)
    logger.info(
        "[FINAL TEST RESULT] Loss: %.4f | Accuracy: %.2f%%",
        test_loss,test_acc*100
    )

def parse_args():
    parser = argparse.ArgumentParser(description='Stanford Dogs Classifier')
    parser.add_argument('--batch_size', help='Batch_size', default=BATCH_SIZE, type=int)
    parser.add_argument('--n_epochs', help='Number of Epochs', default=EPOCHS, type=int)
    parser.add_argument('--learning_rate', help='Learning Rate', default=LR, type=float)
    parser.add_argument('--num_workers', help = 'Number of Workers', default=NUM_WORKERS, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    configure_logging(logfile_path= LOGS_DIR / "train.log")
    main(args= args)