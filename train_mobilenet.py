import os
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import torch_directml
from tqdm import tqdm


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

    def __init__(self, brightness_threshold=225, probability=0.7, noise_mode=True):
        self.brightness_threshold = brightness_threshold
        self.probability = probability
        self.noise_mode = noise_mode

    def __call__(self, img):
        if random.random() > self.probability:
            return img  # μερικές φορές αφήνουμε το αρχικό λευκό φόντο ως έχει

        img_array = np.array(img.convert("RGB"))
        # "Λευκό" pixel: και τα 3 κανάλια πάνω από το κατώφλι
        is_background = np.all(img_array > self.brightness_threshold, axis=-1)

        if self.noise_mode:
            # Τυχαίος θόρυβος - μιμείται "ακατάστατα" φόντα πραγματικών φωτογραφιών
            new_background = np.random.randint(0, 256, img_array.shape, dtype=np.uint8)
        else:
            # Συμπαγές τυχαίο χρώμα
            random_color = np.random.randint(0, 256, size=3, dtype=np.uint8)
            new_background = np.tile(random_color, (img_array.shape[0], img_array.shape[1], 1))

        result = img_array.copy()
        result[is_background] = new_background[is_background]
        return Image.fromarray(result)


# 1. Ρύθμιση AMD GPU μέσω DirectML
if torch_directml.is_available():
    device = torch_directml.device()
    print(f"--> Χρήση AMD GPU: {torch_directml.device_name(0)}")
else:
    device = torch.device("cpu")
    print("--> Η GPU δεν βρέθηκε, χρήση CPU.")

# 2. Hyperparameters & Paths
DATA_DIR = r"C:\Users\crish\Desktop\ptyxiakh\dataset"
BATCH_SIZE = 16  # Μειωμένο από 32: το torch-directml σε AMD GPU δεν ελευθερώνει
                  # καλά τη μνήμη μεταξύ epochs (γνωστό memory leak), οπότε
                  # μικρότερο batch size μειώνει την πίεση στη VRAM
EPOCHS = 10  # Λιγότερα epochs: συνεχίζουμε από ήδη καλά εκπαιδευμένο μοντέλο,
             # δεν ξεκινάμε από την αρχή - αύξησε το αν θες περισσότερη εκπαίδευση
BACKBONE_LR = 0.0002   # Χαμηλότερο από πριν - μικρές, προσεκτικές διορθώσεις
                       # πάνω σε ήδη καλά βάρη, όχι εκπαίδευση από το μηδέν
CLASSIFIER_LR = 0.002
IMAGE_SIZE = 224

# 3. Data Augmentation & Normalization
# Το Fruits-360 έχει πολύ "καθαρές" εικόνες (λευκό φόντο, σταθερός φωτισμός,
# περιστρεφόμενη βάση). Χρησιμοποιούμε πιο επιθετικό augmentation, ΚΑΙ τώρα
# background randomization, ώστε το μοντέλο να μη μάθει το συγκεκριμένο "στυλ"
# του dataset αλλά πιο γενικά χαρακτηριστικά του ίδιου του φρούτου.
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        RandomBackgroundReplace(probability=0.7),
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

# 4. Φόρτωση Δεδομένων
image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
    for x in ['train', 'test']
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    for x in ['train', 'test']
}

class_names = image_datasets['train'].classes
num_classes = len(class_names)
print(f"--> Βρέθηκαν {num_classes} κλάσεις φρούτων!")

# 5. Φόρτωση MobileNetV2 (Fine-Tuning)
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
# να ξανατρέξεις όλα τα epochs από την αρχή για μικρές ρυθμίσεις όπως αυτή.
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

# 6. Loss Function & Optimizer (Χρήση SGD για 100% συμβατότητα με DirectML)
# ΣΗΜΑΝΤΙΚΟ: Πριν υπήρχαν ΔΥΟ optimizer objects — ο δεύτερος αντικαθιστούσε τον
# πρώτο, με αποτέλεσμα τα ξεπαγωμένα blocks του backbone να ΜΗΝ εκπαιδεύονται ποτέ
# (μόνο ο classifier ενημερωνόταν). Εδώ φτιάχνουμε ΕΝΑΝ optimizer με δύο param
# groups, ώστε backbone και classifier να εκπαιδεύονται μαζί, με διαφορετικό LR.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([
    {'params': model.features[-3:].parameters(), 'lr': BACKBONE_LR},
    {'params': model.classifier.parameters(), 'lr': CLASSIFIER_LR},
], momentum=0.9)

# Μειώνει το LR κατά 10x κάθε 7 epochs, βοηθάει τη σύγκλιση στα τελευταία epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 7. Training Loop με Progress Bar (tqdm)
print("\n--- Έναρξη Εκπαίδευσης ---")
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 30)

    for phase in ['train', 'test']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        # Μπάρα προόδου
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

            # Ενημέρωση μπάρας προόδου
            loop.set_postfix(loss=loss.item())

            # Το torch-directml δεν ελευθερώνει πάντα σωστά τη μνήμη GPU μόνο του·
            # σβήνουμε ρητά τους μεγάλους tensors κάθε iteration να βοηθήσουμε τον GC.
            del outputs, inputs, labels, loss, preds

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        if phase == 'train':
            scheduler.step()

    # Αποθήκευση checkpoint ΜΕΤΑ από κάθε epoch (όχι μόνο στο τέλος).
    # Έτσι, αν ξανακολλήσει η μνήμη GPU σε επόμενο epoch, δεν χάνεις την πρόοδο -
    # έχεις ήδη το μοντέλο έως το τελευταίο ολοκληρωμένο epoch αποθηκευμένο.
    SAVE_PATH = "fruit_mobilenetv2.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'completed_epoch': epoch + 1,
    }, SAVE_PATH)
    print(f"--> Checkpoint αποθηκεύτηκε (μετά το epoch {epoch + 1})")

    # Καθαρισμός μνήμης μεταξύ epochs - βοηθάει να καθυστερήσει/αποφευχθεί
    # το memory leak του torch-directml
    gc.collect()

print(f"\n--> Η εκπαίδευση ολοκληρώθηκε! Το μοντέλο αποθηκεύτηκε ως 'fruit_mobilenetv2.pth'")