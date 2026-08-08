"""
evaluate_model.py

Αξιολογεί το εκπαιδευμένο μοντέλο (fruit_mobilenetv2.pth) πάνω στο πλήρες
test set και αποθηκεύει τα αποτελέσματα (συνολική ακρίβεια, per-class ακρίβεια)
στο model_stats.json. Το backend (main.py) διαβάζει αυτό το αρχείο on-demand
για την κάρτα "Στατιστικά Μοντέλου" - δεν τρέχει ξανά το μοντέλο πάνω σε
ολόκληρο το test set σε κάθε request/εκκίνηση server, θα ήταν πολύ αργό.

Τρέξε το ξανά μετά από κάθε νέα εκπαίδευση (train_mobilenet.py) για ενημερωμένα νούμερα:
    python evaluate_model.py
"""
import json
import os
import sys
from datetime import datetime, timezone

# Στο Windows, όταν το stdout ανακατευθύνεται σε αρχείο (π.χ. `> log.txt`), η
# προεπιλεγμένη κωδικοποίηση κονσόλας (cp1252) δεν υποστηρίζει ελληνικούς
# χαρακτήρες και ρίχνει UnicodeEncodeError στο print - αναγκάζουμε UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

DATA_DIR = "dataset"
CHECKPOINT_PATH = "fruit_mobilenetv2.pth"
STATS_PATH = "model_stats.json"
IMAGE_SIZE = 224
BATCH_SIZE = 32


def main():
    device = torch.device("cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = models.mobilenet_v2()
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Το ImageFolder ταξινομεί τις κλάσεις αλφαβητικά με βάση τα ονόματα φακέλων -
    # ίδια λογική με το train_mobilenet.py κατά την εκπαίδευση, οπότε τα class
    # indices ταιριάζουν 1-1 με το checkpoint.
    if test_dataset.classes != class_names:
        raise RuntimeError(
            "Οι κλάσεις του test set δεν ταιριάζουν με τις κλάσεις του checkpoint - "
            "πιθανόν άλλαξε το dataset μετά την εκπαίδευση."
        )

    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Αξιολόγηση στο test set"):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for label, pred in zip(labels, preds):
                class_total[label.item()] += 1
                if label.item() == pred.item():
                    class_correct[label.item()] += 1

    overall_accuracy = round(100 * correct / total, 2)
    per_class_accuracy = {
        class_names[i]: (round(100 * class_correct[i] / class_total[i], 2) if class_total[i] > 0 else None)
        for i in range(num_classes)
    }

    stats = {
        "test_accuracy": overall_accuracy,
        "test_images_evaluated": total,
        "per_class_accuracy": per_class_accuracy,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n--> Ακρίβεια στο test set: {overall_accuracy}% ({correct}/{total})")
    print(f"--> Αποθηκεύτηκε στο {STATS_PATH}")


if __name__ == "__main__":
    main()
