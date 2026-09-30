import os
import gc
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from PIL import Image
import torch_directml
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # χωρίς GUI backend - τρέχει από τερματικό, μόνο αποθήκευση σε αρχείο
import matplotlib.pyplot as plt


class RandomBackgroundReplace:
    """
    Οι εικόνες Fruits-360 έχουν όλες σχεδόν λευκό/ανοιχτόχρωμο φόντο, με σταθερό
    στυλ φωτισμού/σκίασης στις γωνίες. Το Grad-CAM έδειξε ότι το μοντέλο "κολλάει"
    σε αυτό το φόντο αντί να μαθαίνει το ίδιο το φρούτο.
    Αυτό το transform εντοπίζει τα ανοιχτόχρωμα ("λευκά") pixels μιας εικόνας
    (κατώφλι φωτεινότητας) και τα αντικαθιστά με τυχαίο συμπαγές χρώμα ή θόρυβο,
    ΔΙΑΦΟΡΕΤΙΚΟ σε κάθε επανάληψη εκπαίδευσης. Έτσι το μοντέλο δεν μπορεί να
    βασιστεί στο φόντο - αναγκάζεται να μάθει χαρακτηριστικά του ίδιου του φρούτου.
    """

    def __init__(self, brightness_threshold=225, probability=0.5, noise_mode=None):
        self.brightness_threshold = brightness_threshold
        self.probability = probability
        # noise_mode=None: επιλέγεται τυχαία (noise ή συμπαγές χρώμα) σε ΚΑΘΕ κλήση,
        # αντί να είναι πάντα θόρυβος - λιγότερο ακραίο/μονότονο φόντο, πιο κοντά στην
        # ποικιλία πραγματικών φωτογραφιών (π.χ. καθαρό έγχρωμο studio φόντο).
        self.noise_mode = noise_mode

    def __call__(self, img):
        if random.random() > self.probability:
            return img  # μερικές φορές αφήνουμε το αρχικό λευκό φόντο ως έχει

        img_array = np.array(img.convert("RGB"))
        # "Λευκό" pixel: και τα 3 κανάλια πάνω από το κατώφλι
        is_background = np.all(img_array > self.brightness_threshold, axis=-1)

        use_noise = self.noise_mode if self.noise_mode is not None else random.random() < 0.5
        if use_noise:
            # Τυχαίος θόρυβος - μιμείται "ακατάστατα" φόντα πραγματικών φωτογραφιών
            new_background = np.random.randint(0, 256, img_array.shape, dtype=np.uint8)
        else:
            # Συμπαγές τυχαίο χρώμα
            random_color = np.random.randint(0, 256, size=3, dtype=np.uint8)
            new_background = np.tile(random_color, (img_array.shape[0], img_array.shape[1], 1))

        result = img_array.copy()
        result[is_background] = new_background[is_background]
        return Image.fromarray(result)


def update_learning_curve_plot(csv_path, plot_path):

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    test_loss = [float(r["test_loss"]) for r in rows]
    train_acc = [float(r["train_acc"]) * 100 for r in rows]
    test_acc = [float(r["test_acc"]) * 100 for r in rows]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, train_loss, marker="o", label="Train")
    ax_loss.plot(epochs, test_loss, marker="o", label="Test")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Καμπύλη Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, train_acc, marker="o", label="Train")
    ax_acc.plot(epochs, test_acc, marker="o", label="Test")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Καμπύλη Accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.suptitle("Καμπύλες Εκπαίδευσης - MobileNetV2 Fruit Classifier")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


if torch_directml.is_available():
    device = torch_directml.device()
    print(f"--> Χρήση AMD GPU: {torch_directml.device_name(0)}")
else:
    device = torch.device("cpu")
    print("--> Η GPU δεν βρέθηκε, χρήση CPU.")

DATA_DIR = r"C:\Users\crish\Desktop\ptyxiakh\dataset"
BATCH_SIZE = 16 # μικρότερο batch size μειώνει την πίεση στη VRAM
EPOCHS = 30
BACKBONE_LR = 0.0002
CLASSIFIER_LR = 0.002
IMAGE_SIZE = 224

# Χρησιμοποιούμε πιο επιθετικό augmentation, και τώρα
# background randomization, ώστε το μοντέλο να μη μάθει το συγκεκριμένο "στυλ"
# του dataset αλλά πιο γενικά χαρακτηριστικά του ίδιου του φρούτου.
#
# ΠΡΟΣΟΧΗ: saturation/brightness/contrast=0.3 + RandomBackgroundReplace(0.7, πάντα
# θόρυβος) αποδείχτηκε ΥΠΕΡΒΟΛΙΚΟ - σε δοκιμές με "γυαλιστερές" stock φωτογραφίες
# (καθαρό λευκό φόντο, φυσικό φως) το μοντέλο μπέρδευε π.χ. ανανά με βατόμουρο και
# καράμβολα με μπανάνα, σαν να είχε μάθει να βασίζεται σε σχήμα/υφή αντί για χρώμα.
# Πιο ήπιο εδώ (λιγότερο έντονο color jitter, background replacement λιγότερο συχνά
# και όχι πάντα θόρυβος - βλ. RandomBackgroundReplace) ώστε να μείνει η ανθεκτικότητα
# στο φόντο χωρίς να θυσιάζεται το χρώμα ως discriminative χαρακτηριστικό.
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        RandomBackgroundReplace(probability=0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), data_transforms['train'])
class_names = train_dataset.classes
num_classes = len(class_names)
print(f"--> Βρέθηκαν {num_classes} κλάσεις φρούτων!")

test_dataset_raw = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), data_transforms['test'])
name_to_class_idx = {name: i for i, name in enumerate(class_names)}
unmatched = [c for c in test_dataset_raw.classes if c not in name_to_class_idx]
if unmatched:
    print(f"ΠΡΟΣΟΧΗ: {len(unmatched)} φάκελοι του 'test' δεν αντιστοιχούν σε καμία κλάση "
            f"του 'train' - εξαιρούνται από την αξιολόγηση: {', '.join(unmatched)}")
raw_idx_to_class_idx = {
    i: name_to_class_idx[name] for i, name in enumerate(test_dataset_raw.classes) if name in name_to_class_idx
}
test_matched_indices = [i for i, t in enumerate(test_dataset_raw.targets) if t in raw_idx_to_class_idx]
test_dataset_raw.target_transform = lambda t: raw_idx_to_class_idx[t]
test_dataset = Subset(test_dataset_raw, test_matched_indices)

image_datasets = {'train': train_dataset, 'test': test_dataset}

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
    'test': DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
}

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Ξεπαγώνουμε τα τελευταία 3 blocks του MobileNetV2 για να μάθει καλύτερα τα σχήματα
for param in model.features[-3:].parameters():
    param.requires_grad = True

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Linear(in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, num_classes)
)

# Αν υπάρχει ήδη ένα εκπαιδευμένο μοντέλο από προηγούμενο τρέξιμο, ΣΥΝΕΧΙΖΟΥΜΕ
# από εκεί αντί να ξεκινήσουμε από τα αρχικά ImageNet βάρη. Έτσι δεν χρειάζεται
# να ξανατρέξεις όλα τα epochs από την αρχή.
CHECKPOINT_PATH = "fruit_mobilenetv2.pth"
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if checkpoint.get("class_names") == class_names:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"--> Βρέθηκε υπάρχον checkpoint· συνεχίζουμε την εκπαίδευση από εκεί "
                f"(προηγούμενο epoch: {checkpoint.get('completed_epoch', '?')}).")
    else:
        print("--> Βρέθηκε checkpoint αλλά με διαφορετικές κλάσεις (π.χ. άλλαξες dataset)· "
                "ξεκινάμε από την αρχή με τα ImageNet βάρη.")
else:
    print("--> Δεν βρέθηκε υπάρχον checkpoint· ξεκινάμε από την αρχή με τα ImageNet βάρη.")

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([
    {'params': model.features[-3:].parameters(), 'lr': BACKBONE_LR},
    {'params': model.classifier.parameters(), 'lr': CLASSIFIER_LR},
], momentum=0.9)

# Μειώνει το LR κατά 10x κάθε 7 epochs, βοηθάει τη σύγκλιση στα τελευταία epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Learning curve logging
LOG_CSV_PATH = "training_log.csv"
PLOT_PATH = "training_curves.png"

log_already_exists = os.path.exists(LOG_CSV_PATH)
starting_epoch_number = 1
if log_already_exists:
    with open(LOG_CSV_PATH, "r", newline="", encoding="utf-8") as f:
        existing_rows = list(csv.reader(f))
    if len(existing_rows) > 1:  # header + τουλάχιστον ένα epoch
        starting_epoch_number = int(existing_rows[-1][0]) + 1
else:
    with open(LOG_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])

print("\n--- Έναρξη Εκπαίδευσης ---")
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 30)

    epoch_metrics = {}

    for phase in ['train', 'test']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        loop = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Phase", leave=False)

        for inputs, labels in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            loop.set_postfix(loss=loss.item())

            del outputs, inputs, labels, loss, preds
        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])
        epoch_metrics[phase] = (epoch_loss, float(epoch_acc))
        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        if phase == 'train':
            scheduler.step()

    # Καταγραφή καμπυλών εκπαίδευσης (CSV + PNG) ΜΕΤΑ από κάθε epoch
    global_epoch = starting_epoch_number + epoch
    train_loss, train_acc = epoch_metrics['train']
    test_loss, test_acc = epoch_metrics['test']
    with open(LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([global_epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
                            f"{test_loss:.4f}", f"{test_acc:.4f}"])
    update_learning_curve_plot(LOG_CSV_PATH, PLOT_PATH)

    # Αποθήκευση checkpoint ΜΕΤΑ από κάθε epoch (όχι μόνο στο τέλος).
    SAVE_PATH = "fruit_mobilenetv2.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'completed_epoch': epoch + 1,
    }, SAVE_PATH)
    print(f"--> Checkpoint αποθηκεύτηκε (μετά το epoch {epoch + 1})")

    # Καθαρισμός μνήμης μεταξύ epochs
    gc.collect()

print(f"\n--> Η εκπαίδευση ολοκληρώθηκε! Το μοντέλο αποθηκεύτηκε ως 'fruit_mobilenetv2.pth'")