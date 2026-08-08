import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from collections import Counter
import time

# 1. Φόρτωση Μοντέλου
MODEL_PATH = "fruit_mobilenetv2.pth"
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)

class_names = checkpoint['class_names']
num_classes = len(class_names)

model = models.mobilenet_v2()
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Linear(in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, num_classes)
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. Transform για το AI
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Εκκίνηση Κάμερας
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Σφάλμα: Δεν ήταν δυνατή η πρόσβαση στην κάμερα (index 0).")
    print("Ελέγξτε ότι είναι συνδεδεμένη και δεν τη χρησιμοποιεί ήδη άλλη εφαρμογή.")
    raise SystemExit(1)

WINDOW_NAME = 'AI Fruit Detection with Bounding Boxes'
# WINDOW_NORMAL κάνει το παράθυρο πραγματικά resizable, ώστε η εικόνα να
# τεντώνεται και να γεμίζει σωστά όταν το μεγιστοποιείς (fullscreen)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

print("=" * 55)
print(" AI Αναγνώριση Φρούτων - Ζωντανή Κάμερα")
print("=" * 55)
print("Οδηγίες χρήσης:")
print(" - Κράτησε ένα φρούτο μπροστά στην κάμερα, σε φόντο")
print("   που να ξεχωρίζει από το χρώμα του (π.χ. όχι έντονα")
print("   χρωματιστό ή ίδιου χρώματος με το φρούτο).")
print(" - Το πράσινο πλαίσιο δείχνει τι έχει εντοπιστεί.")
print(" - Το όνομα του φρούτου και το ποσοστό βεβαιότητας")
print("   ανανεώνονται περίπου κάθε 2 δευτερόλεπτα.")
print(" - Αν η βεβαιότητα είναι πολύ χαμηλή, εμφανίζεται")
print("   'Not recognized' αντί για πιθανώς λάθος όνομα.")
print(" - Πίεσε 'q' για έξοδο από το πρόγραμμα.")
print("=" * 55)

# Η ετικέτα ανανεώνεται μία φορά κάθε UPDATE_INTERVAL frames (όχι σε κάθε frame),
# χρησιμοποιώντας την πιο συχνή πρόβλεψη μέσα σε αυτό το διάστημα
UPDATE_INTERVAL = 60
# Κάτω από αυτό το confidence (0.0-1.0), η πρόβλεψη θεωρείται αναξιόπιστη και
# εμφανίζεται "Not recognized" αντί για ένα πιθανώς λάθος όνομα φρούτου
MIN_CONFIDENCE = 0.40
# Ελάχιστος μέσος κορεσμός (0-255) μέσα στο υποψήφιο πλαίσιο ώστε να θεωρείται
# πιθανό αντικείμενο. Το Otsu threshold είναι ΣΧΕΤΙΚΟ (χωρίζει το frame στα 2
# πιο "διαχωρίσιμα" group ό,τι κι αν υπάρχει μπροστά στην κάμερα) - χωρίς αυτόν
# τον απόλυτο έλεγχο, ακόμα και ένα άδειο δωμάτιο μπορεί να παράξει "candidate"
# απλά επειδή κάποιο σημείο τυχαίνει να είναι σχετικά πιο κορεσμένο απ' τα άλλα
MIN_SATURATION = 60
# Ελάχιστο ποσοστό "γεμίσματος" του bounding rect από το ίδιο το contour
# (area / εμβαδόν rect). Ένα συμπαγές φρούτο γεμίζει καλά το ορθογώνιό του,
# ενώ ένα blob που "κόλλησε" με reflection/έπιπλο μέσω του morphological
# closing απλώνεται σε πολύ μεγαλύτερο (και αραιά γεμάτο) bounding rect
MIN_EXTENT = 0.4
# Πόσο γρήγορα το πλαίσιο "κυνηγάει" τη νέα θέση κάθε frame (0-1): μικρότερο =
# πιο σταθερό αλλά πιο αργό να ακολουθήσει, μεγαλύτερο = πιο γρήγορο αλλά πιο
# ασταθές/τρεμοπαίζον
BOX_SMOOTHING = 0.3
# Μετά από τόσα συνεχόμενα frames χωρίς έγκυρη ανίχνευση, το πλαίσιο "ξεχνάει"
# την τελευταία θέση αντί να μείνει κολλημένο εκεί
MAX_MISSED_FRAMES = 10
frame_count = 0
prediction_buffer = []
displayed_fruit_name = None
displayed_confidence = 0.0
smoothed_box = None
missed_frames = 0

# Το FPS υπολογίζεται με εκθετικό κινητό μέσο όρο (EMA) ώστε η ένδειξη στην
# οθόνη να μην τρεμοπαίζει σε κάθε frame
fps = 0.0
prev_frame_time = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- ΕΝΤΟΠΙΣΜΟΣ ΠΕΡΙΓΡΑΜΜΑΤΟΣ (BOUNDING BOX DETECTION) ---
        # Χρησιμοποιούμε το κανάλι κορεσμού (saturation) του HSV: τα φρούτα έχουν
        # έντονο χρώμα ενώ τοίχοι/έπιπλα είναι συνήθως "ξεθωριασμένα" (χαμηλός κορεσμός),
        # οπότε ξεχωρίζουν πιο αξιόπιστα απ' ό,τι με απλή φωτεινότητα (gray).
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        blurred = cv2.GaussianBlur(saturation, (7, 7), 0)

        # Thresholding (Otsu: προσαρμόζεται αυτόματα στις συνθήκες)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Καθαρισμός θορύβου: πρώτα opening (σπάει λεπτές "γέφυρες" σύνδεσης με το
        # φόντο, ώστε το πλαίσιο να μη διογκώνεται), μετά closing (γεμίζει μικρές
        # τρύπες μέσα στο ίδιο το αντικείμενο, π.χ. από αντανακλάσεις φωτός)
        open_kernel = np.ones((9, 9), np.uint8)
        close_kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = frame.shape[0] * frame.shape[1]
        object_detected = False

        def contour_extent(c):
            _, _, cw, ch = cv2.boundingRect(c)
            rect_area = cw * ch
            return cv2.contourArea(c) / rect_area if rect_area > 0 else 0

        # Κρατάμε όλα τα περιγράμματα με λογικό μέγεθος ΚΑΙ σχήμα για αντικείμενο
        # (όχι θόρυβος, όχι ολόκληρο το φόντο/χέρι, όχι blob κολλημένο σε reflection -
        # βλ. σχόλιο στο MIN_EXTENT)
        candidates = [
            c for c in contours
            if 2000 < cv2.contourArea(c) < 0.7 * frame_area and contour_extent(c) > MIN_EXTENT
        ]

        if candidates:
            # Συνδυάζουμε εμβαδόν ΚΑΙ απόσταση από το κέντρο σε ένα σκορ: ούτε το
            # απλά μεγαλύτερο contour (μπορεί να είναι φόντο), ούτε το απλά πιο
            # κεντρικό (μπορεί να είναι μικρή ανταύγεια/θόρυβος) είναι από μόνο
            # του αξιόπιστο. Προτιμάμε μεγάλα αντικείμενα κοντά στο κέντρο.
            frame_cx, frame_cy = frame.shape[1] / 2, frame.shape[0] / 2
            max_dist = ((frame.shape[1] / 2) ** 2 + (frame.shape[0] / 2) ** 2) ** 0.5

            def score(c):
                area = cv2.contourArea(c)
                cx, cy, cw, ch = cv2.boundingRect(c)
                center_x, center_y = cx + cw / 2, cy + ch / 2
                dist = ((center_x - frame_cx) ** 2 + (center_y - frame_cy) ** 2) ** 0.5
                normalized_dist = dist / max_dist  # 0 (κέντρο) έως ~1 (γωνία)
                return area * (1 - normalized_dist)

            # Δοκιμάζουμε τα υποψήφια με σειρά προτεραιότητας (μεγαλύτερο score πρώτο)
            # και κρατάμε το ΠΡΩΤΟ που περνάει και τον έλεγχο απόλυτου κορεσμού - όχι
            # μόνο το κορυφαίο. Αλλιώς, αν το κορυφαίο (π.χ. πιο κεντρικό αλλά στην
            # ουσία φόντο) αποτύχει στο κορεσμό, χανόταν εντελώς η ανίχνευση αντί να
            # δοκιμαστεί το επόμενο καλύτερο υποψήφιο (π.χ. το πραγματικό φρούτο,
            # έστω πιο μακριά από το κέντρο του frame)
            for c in sorted(candidates, key=score, reverse=True):
                cx, cy, cw, ch = cv2.boundingRect(c)
                mean_saturation = saturation[cy:cy+ch, cx:cx+cw].mean()
                if mean_saturation >= MIN_SATURATION:
                    bx, by, bw, bh = cx, cy, cw, ch
                    object_detected = True
                    break

        if object_detected:
            # Εξομάλυνση του πλαισίου μέσω EMA (exponential moving average) στις
            # συντεταγμένες, ώστε να μην τρεμοπαίζει από frame σε frame λόγω
            # μικρών αλλαγών στο contour (θόρυβος φωτισμού κ.λπ.)
            if smoothed_box is None:
                smoothed_box = [float(bx), float(by), float(bw), float(bh)]
            else:
                smoothed_box[0] += (bx - smoothed_box[0]) * BOX_SMOOTHING
                smoothed_box[1] += (by - smoothed_box[1]) * BOX_SMOOTHING
                smoothed_box[2] += (bw - smoothed_box[2]) * BOX_SMOOTHING
                smoothed_box[3] += (bh - smoothed_box[3]) * BOX_SMOOTHING
            missed_frames = 0

            frame_h, frame_w = frame.shape[:2]
            x, y, w, h = (int(v) for v in smoothed_box)
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = max(1, min(w, frame_w - x))
            h = max(1, min(h, frame_h - y))

            # Αποκοπή της περιοχής του φρούτου (Crop ROI)
            roi = frame[y:y+h, x:x+w]
            if roi.size > 0:
                # Μετατροπή BGR σε RGB για το PyTorch
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_roi)

                # AI Inference
                input_tensor = transform(pil_img).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs[0], dim=0)
                    conf, pred_idx = torch.max(probs, 0)

                # Συλλέγουμε τις προβλέψεις του διαστήματος· η εμφανιζόμενη
                # ετικέτα θα ανανεωθεί μόνο κάθε UPDATE_INTERVAL frames
                prediction_buffer.append((pred_idx.item(), conf.item()))

                # --- ΣΧΕΔΙΑΣΗ BOUNDING BOX & LABEL ---
                # Πράσινο Πλαίσιο (ακολουθεί το αντικείμενο σε κάθε frame)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                if displayed_fruit_name is not None:
                    if displayed_fruit_name == "Not recognized":
                        label_text = "Not recognized"
                    else:
                        label_text = f"{displayed_fruit_name} ({displayed_confidence:.1f}%)"
                    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

                    # Αν δεν χωράει η ετικέτα πάνω από το πλαίσιο, τη βάζουμε από κάτω
                    label_y_top = y - text_h - 10
                    if label_y_top < 0:
                        label_y_top = y + h
                    label_y_bottom = label_y_top + text_h + 10
                    label_y_text = label_y_top + text_h + 5

                    cv2.rectangle(frame, (x, label_y_top), (x + text_w, label_y_bottom), (0, 255, 0), -1)

                    # Κείμενο πάνω/κάτω από το Bounding Box
                    cv2.putText(frame, label_text, (x, label_y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        if not object_detected:
            # Λίγα διαδοχικά χαμένα frames είναι φυσιολογικά (π.χ. στιγμιαία κίνηση) -
            # δεν πετάμε αμέσως το smoothed_box, μόνο αν επιμείνει η απουσία ανίχνευσης
            missed_frames += 1
            if missed_frames > MAX_MISSED_FRAMES:
                smoothed_box = None

        # Μία φορά κάθε UPDATE_INTERVAL frames, ανανεώνουμε την εμφανιζόμενη ετικέτα
        # με βάση την πιο συχνή πρόβλεψη που συγκεντρώθηκε στο διάστημα αυτό
        frame_count += 1
        if frame_count % UPDATE_INTERVAL == 0:
            if prediction_buffer:
                best_idx, count = Counter(idx for idx, _ in prediction_buffer).most_common(1)[0]
                avg_conf = sum(c for idx, c in prediction_buffer if idx == best_idx) / count
                if avg_conf >= MIN_CONFIDENCE:
                    displayed_fruit_name = class_names[best_idx].replace("_", " ")
                else:
                    displayed_fruit_name = "Not recognized"
                displayed_confidence = avg_conf * 100
            prediction_buffer = []

        # --- ΥΠΟΛΟΓΙΣΜΟΣ FPS ---
        # Εκθετικός κινητός μέσος όρος (EMA) ώστε η ένδειξη να μην τρεμοπαίζει
        # σε κάθε frame
        current_time = time.time()
        if prev_frame_time is not None:
            elapsed = current_time - prev_frame_time
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0.0 else (fps * 0.9 + instant_fps * 0.1)
        prev_frame_time = current_time

        # --- ΜΠΑΡΑ ΟΔΗΓΙΩΝ / ΚΑΤΑΣΤΑΣΗΣ (πάνω μέρος οθόνης) ---
        # Σημείωση: η OpenCV δεν αποδίδει σωστά ελληνικούς χαρακτήρες με putText,
        # γι' αυτό το on-screen κείμενο είναι στα αγγλικά.
        status_text = "Analyzing fruit..." if object_detected else "Show a fruit to the camera"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        fps_text = f"FPS: {fps:.1f}"
        (fps_w, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, fps_text, (frame.shape[1] - fps_w - 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Το WINDOW_NORMAL τεντώνει την εικόνα να γεμίσει όλο το παράθυρο, κάτι
        # που τη διαστρεβλώνει όταν το παράθυρο έχει διαφορετικές αναλογίες
        # (π.χ. σε fullscreen/maximize). Εδώ κρατάμε τις σωστές αναλογίες και
        # προσθέτουμε μαύρες λωρίδες (letterboxing) γύρω από την εικόνα.
        win_x, win_y, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
        if win_w > 0 and win_h > 0:
            frame_h, frame_w = frame.shape[:2]
            scale = min(win_w / frame_w, win_h / frame_h)
            new_w, new_h = max(1, int(frame_w * scale)), max(1, int(frame_h * scale))
            resized = cv2.resize(frame, (new_w, new_h))

            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            x_off, y_off = (win_w - new_w) // 2, (win_h - new_h) // 2
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
            display_frame = canvas
        else:
            display_frame = frame

        cv2.imshow(WINDOW_NAME, display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Αν ο χρήστης κλείσει το παράθυρο με το X, το WND_PROP_VISIBLE πέφτει
        # στο 0 - χωρίς αυτόν τον έλεγχο ο βρόχος θα συνέχιζε να τρέχει στο κενό
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
finally:
    # Εξασφαλίζει ότι η κάμερα ελευθερώνεται και τα παράθυρα κλείνουν ακόμα κι
    # αν πεταχτεί exception μέσα στο βρόχο
    cap.release()
    cv2.destroyAllWindows()
