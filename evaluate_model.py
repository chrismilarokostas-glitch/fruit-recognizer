"""
evaluate_model.py

Τρέχει το εκπαιδευμένο μοντέλο πάνω στο test set και βρίσκει σε ΠΟΙΑ φρούτα
κάνει τα περισσότερα λάθη, καθώς και ΜΕ ΠΟΙΑ άλλα φρούτα τα συγχέει πιο συχνά
(confusion matrix). Ο στόχος είναι να δείξει ποιες κλάσεις χρειάζονται
ενίσχυση (περισσότερα/καλύτερα δεδομένα, augmentation) στο επόμενο training.

Χρήση:
    python evaluate_model.py
    python evaluate_model.py --split test --top 20 --csv report.csv
    python evaluate_model.py --model fruit_mobilenetv2.pth --data-dir dataset
"""

import argparse
import csv
import os
import sys

# Σε Windows, η κονσόλα χρησιμοποιεί συχνά cp1252 by default, που δεν
# υποστηρίζει ελληνικούς χαρακτήρες - χωρίς αυτό, τα print() με ελληνικό
# κείμενο (π.χ. στο get_device()) πετάνε UnicodeEncodeError και το script
# κρασάρει πριν καν ξεκινήσει η αξιολόγηση.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

IMAGE_SIZE = 224


def get_device():
    # Ίδιο pattern με το train_mobilenet.py: χρήση AMD GPU μέσω DirectML αν
    # υπάρχει, αλλιώς CPU. Το import είναι προαιρετικό εδώ (σε αντίθεση με το
    # training script) ώστε αυτό το evaluation script να τρέχει και σε μηχάνημα
    # χωρίς torch_directml εγκατεστημένο.
    try:
        import torch_directml
        if torch_directml.is_available():
            print(f"--> Χρήση AMD GPU: {torch_directml.device_name(0)}")
            return torch_directml.device()
    except ImportError:
        pass
    print("--> Χρήση CPU.")
    return torch.device("cpu")


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = models.mobilenet_v2()
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names


def warn_if_train_folder_out_of_sync(data_dir, class_names):
    """
    Το checkpoint αποθηκεύει τις κλάσεις όπως ήταν το dataset/train ΤΗΝ ΩΡΑ
    της εκπαίδευσης. Αν ο φάκελος train άλλαξε έκτοτε (μετονομασία/προσθήκη/
    αφαίρεση φακέλου φρούτου), ένα μελλοντικό resume-training θα αποτύχει στον
    έλεγχο `checkpoint['class_names'] == class_names` και θα ξεκινήσει ΕΞ ΑΡΧΗΣ
    από τα ImageNet βάρη αντί να συνεχίσει - χάνοντας σιωπηλά όλη την πρόοδο.
    """
    train_dir = os.path.join(data_dir, "train")
    if not os.path.isdir(train_dir):
        return

    train_classes = sorted(
        d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
    )
    if sorted(class_names) == train_classes:
        return

    only_in_model = sorted(set(class_names) - set(train_classes))
    only_in_train = sorted(set(train_classes) - set(class_names))

    print("=" * 70)
    print("ΠΡΟΣΟΧΗ: το 'dataset/train' δεν ταιριάζει πλέον με τις κλάσεις του μοντέλου.")
    print("Αν ξανατρέξεις το train_mobilenet.py τώρα, ο έλεγχος συνέχισης θα αποτύχει")
    print("και θα ξεκινήσει ΕΞ ΑΡΧΗΣ από τα ImageNet βάρη (χάνοντας την τρέχουσα εκπαίδευση).")
    if only_in_model:
        print(f"  Κλάσεις μοντέλου που ΔΕΝ υπάρχουν πια στο train: {', '.join(only_in_model)}")
    if only_in_train:
        print(f"  Φάκελοι στο train που δεν αντιστοιχούν σε καμία κλάση μοντέλου: {', '.join(only_in_train)}")
    print("=" * 70)
    print()


def build_eval_loader(data_dir, split, class_names, batch_size):
    """
    Φορτώνει το split (π.χ. 'test') και αντιστοιχίζει τους φακέλους του στις
    κλάσεις του μοντέλου ΜΕ ΒΑΣΗ ΤΟ ΟΝΟΜΑ (όχι με βάση τη σειρά) - έτσι, αν το
    split έχει φακέλους που δεν υπάρχουν στο μοντέλο (βλ. warn_if_train_folder_out_of_sync
    για ένα ακριβώς τέτοιο πρόβλημα), τους εξαιρούμε ρητά αντί να μπερδέψουμε
    δείγματα σε λάθος κλάση σιωπηλά.
    """
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(os.path.join(data_dir, split), eval_transform)
    name_to_ckpt_idx = {name: i for i, name in enumerate(class_names)}

    unmatched = [c for c in dataset.classes if c not in name_to_ckpt_idx]
    if unmatched:
        print(f"ΠΡΟΣΟΧΗ: {len(unmatched)} φάκελοι του '{split}' δεν αντιστοιχούν σε καμία "
              f"κλάση του μοντέλου - εξαιρούνται από την αξιολόγηση:")
        for c in unmatched:
            print(f"   - {c}")
        print()

    dataset_idx_to_ckpt_idx = {
        i: name_to_ckpt_idx[name] for i, name in enumerate(dataset.classes) if name in name_to_ckpt_idx
    }
    matched_indices = [i for i, t in enumerate(dataset.targets) if t in dataset_idx_to_ckpt_idx]
    excluded = len(dataset) - len(matched_indices)
    if excluded:
        print(f"({excluded} εικόνες εξαιρέθηκαν λόγω μη αντιστοιχισμένων κλάσεων)\n")

    # Ξαναχαρτογραφεί κάθε label από τον δείκτη-του-φακέλου (dataset-space) στον
    # δείκτη-κλάσης του μοντέλου (checkpoint-space) - χρειάζεται ώστε οι
    # προβλέψεις του μοντέλου να συγκρίνονται σωστά με τα πραγματικά labels.
    dataset.target_transform = lambda t: dataset_idx_to_ckpt_idx[t]

    matched_paths = [dataset.samples[i][0] for i in matched_indices]
    subset = Subset(dataset, matched_indices)
    # shuffle=False + num_workers=0: η σειρά επιστροφής είναι ντετερμινιστική
    # και ταιριάζει ακριβώς με τη σειρά του matched_paths - το χρειαζόμαστε
    # για να ξέρουμε ΠΟΙΟ αρχείο αντιστοιχεί σε ποια πρόβλεψη μετά το loop.
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, matched_paths


def run_inference(model, loader, device):
    all_true, all_pred, all_conf = [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Αξιολόγηση", unit="batch"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, dim=1)

            all_true.extend(labels.tolist())
            all_pred.extend(preds.cpu().tolist())
            all_conf.extend(conf.cpu().tolist())
    return np.array(all_true), np.array(all_pred), np.array(all_conf)


def compute_per_class_stats(class_names, all_true, all_pred, all_conf, confusion, top_confusions):
    stats = []
    for idx, name in enumerate(class_names):
        mask = all_true == idx
        total = int(mask.sum())
        if total == 0:
            stats.append({"name": name, "total": 0, "correct": 0, "errors": 0,
                           "accuracy": None, "avg_conf_correct": None, "avg_conf_wrong": None,
                           "confusions": []})
            continue

        correct_mask = mask & (all_pred == idx)
        wrong_mask = mask & (all_pred != idx)
        correct = int(correct_mask.sum())
        errors = total - correct

        row = confusion[idx].copy()
        row[idx] = 0  # μας ενδιαφέρουν μόνο τα λάθη, όχι οι σωστές προβλέψεις
        order = np.argsort(row)[::-1]
        confusions = [(class_names[j], int(row[j])) for j in order[:top_confusions] if row[j] > 0]

        stats.append({
            "name": name, "total": total, "correct": correct, "errors": errors,
            "accuracy": correct / total,
            "avg_conf_correct": float(all_conf[correct_mask].mean()) if correct > 0 else None,
            "avg_conf_wrong": float(all_conf[wrong_mask].mean()) if errors > 0 else None,
            "confusions": confusions,
        })
    return stats


def print_report(stats, top_n):
    evaluated = [s for s in stats if s["total"] > 0]
    no_data = [s for s in stats if s["total"] == 0]
    evaluated.sort(key=lambda s: s["accuracy"])

    overall_correct = sum(s["correct"] for s in evaluated)
    overall_total = sum(s["total"] for s in evaluated)
    macro_acc = sum(s["accuracy"] for s in evaluated) / len(evaluated)

    print("=" * 70)
    print(" ΑΠΟΤΕΛΕΣΜΑΤΑ ΑΞΙΟΛΟΓΗΣΗΣ")
    print("=" * 70)
    print(f"Συνολική ακρίβεια (micro):     {overall_correct/overall_total*100:5.2f}%  ({overall_correct}/{overall_total})")
    print(f"Μέση ακρίβεια ανά κλάση (macro): {macro_acc*100:5.2f}%")

    print(f"\nTop {min(top_n, len(evaluated))} φρούτα με τα περισσότερα λάθη "
          f"(ταξινομημένα από τα χειρότερα):")
    print("-" * 70)
    for rank, s in enumerate(evaluated[:top_n], start=1):
        conf_correct = f"{s['avg_conf_correct']*100:.0f}%" if s["avg_conf_correct"] is not None else "-"
        conf_wrong = f"{s['avg_conf_wrong']*100:.0f}%" if s["avg_conf_wrong"] is not None else "-"
        print(f"{rank:2d}. {s['name']:<20} δείγματα={s['total']:<4} λάθη={s['errors']:<4} "
              f"ακρίβεια={s['accuracy']*100:5.1f}%   μέση εμπιστοσύνη (σωστά/λάθος): {conf_correct}/{conf_wrong}")
        if s["confusions"]:
            conf_str = ", ".join(f"{n} ({c}x)" for n, c in s["confusions"])
            print(f"      -> συγχέεται συχνότερα με: {conf_str}")
        else:
            print("      -> κανένα λάθος δεν επαναλαμβάνεται σταθερά με συγκεκριμένο φρούτο")

    if no_data:
        print(f"\n(Δεν βρέθηκαν δείγματα στο split για: {', '.join(s['name'] for s in no_data)})")
    print()


def save_csv_report(stats, path, top_confusions):
    stats_sorted = sorted(
        (s for s in stats if s["total"] > 0),
        key=lambda s: s["accuracy"]
    )
    header = ["fruit", "samples", "correct", "errors", "accuracy_pct",
              "avg_confidence_correct_pct", "avg_confidence_wrong_pct"]
    for i in range(1, top_confusions + 1):
        header += [f"confused_with_{i}", f"confused_with_{i}_count"]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for s in stats_sorted:
            row = [
                s["name"], s["total"], s["correct"], s["errors"],
                round(s["accuracy"] * 100, 2),
                round(s["avg_conf_correct"] * 100, 1) if s["avg_conf_correct"] is not None else "",
                round(s["avg_conf_wrong"] * 100, 1) if s["avg_conf_wrong"] is not None else "",
            ]
            for i in range(top_confusions):
                if i < len(s["confusions"]):
                    row += [s["confusions"][i][0], s["confusions"][i][1]]
                else:
                    row += ["", ""]
            writer.writerow(row)
    print(f"--> Πλήρης αναφορά (όλες οι κλάσεις) αποθηκεύτηκε στο: {path}")


def save_error_examples(class_names, matched_paths, all_true, all_pred, all_conf, stats, top_n, examples_per_class, path):
    """
    Για τις top_n χειρότερες κλάσεις, αποθηκεύει τα λάθη με τη ΜΕΓΑΛΥΤΕΡΗ
    εμπιστοσύνη του μοντέλου (πιο "σίγουρα λάθος" προβλέψεις) - αυτές είναι
    συνήθως οι πιο ενδιαφέρουσες για να δεις τι μπερδεύει το μοντέλο (π.χ.
    οπτικά πολύ όμοιο φρούτο, ή ίδιο ίσως mislabeled δείγμα).
    """
    evaluated = [s for s in stats if s["total"] > 0]
    evaluated.sort(key=lambda s: s["accuracy"])
    worst_names = {s["name"] for s in evaluated[:top_n]}
    name_to_idx = {n: i for i, n in enumerate(class_names)}

    rows = []
    for name in worst_names:
        idx = name_to_idx[name]
        mistake_positions = [
            i for i in range(len(all_true))
            if all_true[i] == idx and all_pred[i] != idx
        ]
        mistake_positions.sort(key=lambda i: all_conf[i], reverse=True)
        for i in mistake_positions[:examples_per_class]:
            rows.append([
                name, class_names[all_pred[i]], round(float(all_conf[i]) * 100, 1), matched_paths[i]
            ])

    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true_fruit", "predicted_as", "model_confidence_pct", "image_path"])
        writer.writerows(rows)
    print(f"--> Παραδείγματα λαθών (top {top_n} χειρότερες κλάσεις) αποθηκεύτηκαν στο: {path}")


def main():
    parser = argparse.ArgumentParser(description="Αξιολόγηση μοντέλου - βρίσκει τα φρούτα με τα περισσότερα λάθη")
    parser.add_argument("--model", default="fruit_mobilenetv2.pth", help="Path στο checkpoint (.pth)")
    parser.add_argument("--data-dir", default="dataset", help="Φάκελος dataset (περιέχει train/ και test/)")
    parser.add_argument("--split", default="test", choices=["test", "train"], help="Ποιο split να αξιολογηθεί")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top", type=int, default=15, help="Πόσα χειρότερα φρούτα να εμφανιστούν στην αναφορά")
    parser.add_argument("--confusions", type=int, default=3, help="Πόσες πιο συχνές συγχύσεις να δείχνει ανά φρούτο")
    parser.add_argument("--csv", default="evaluation_report.csv", help="Path για την πλήρη αναφορά CSV (όλες οι κλάσεις). Βάλε '' για να μην αποθηκευτεί.")
    parser.add_argument("--error-examples-csv", default="evaluation_errors.csv", help="Path για παραδείγματα λαθών (εικόνες) των χειρότερων κλάσεων. Βάλε '' για να μην αποθηκευτεί.")
    parser.add_argument("--examples-per-class", type=int, default=5, help="Πόσα παραδείγματα λαθών ανά χειρότερη κλάση")
    args = parser.parse_args()

    device = get_device()
    model, class_names = load_model(args.model, device)
    print(f"--> Μοντέλο φορτώθηκε: {len(class_names)} κλάσεις\n")

    warn_if_train_folder_out_of_sync(args.data_dir, class_names)

    loader, matched_paths = build_eval_loader(args.data_dir, args.split, class_names, args.batch_size)
    all_true, all_pred, all_conf = run_inference(model, loader, device)

    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_true, all_pred):
        confusion[t, p] += 1

    stats = compute_per_class_stats(class_names, all_true, all_pred, all_conf, confusion, args.confusions)
    print_report(stats, args.top)

    if args.csv:
        save_csv_report(stats, args.csv, args.confusions)
    if args.error_examples_csv:
        save_error_examples(class_names, matched_paths, all_true, all_pred, all_conf,
                             stats, args.top, args.examples_per_class, args.error_examples_csv)


if __name__ == "__main__":
    main()
