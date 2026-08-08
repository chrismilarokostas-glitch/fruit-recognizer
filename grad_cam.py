"""
grad_cam.py

Δείχνει οπτικά ΠΟΥ "κοιτάει" το μοντέλο πάνω σε μια εικόνα για να αποφασίσει
την πρόβλεψή του (Grad-CAM heatmap). Αν το κόκκινο/κίτρινο σημείο πέφτει πάνω
στο φόντο ή στις άκρες αντί στο ίδιο το φρούτο, αυτό επιβεβαιώνει οπτικά ότι
το μοντέλο έμαθε λάθος χαρακτηριστικά (π.χ. το φόντο, όχι το φρούτο).

Χρειάζεται: pip install matplotlib --break-system-packages (αν δεν το έχεις ήδη)

Χρήση:
    python grad_cam.py --image photo.jpg
    python grad_cam.py --image photo.jpg --model fruit_mobilenetv2.pth --out result.png
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.cm as cm


IMAGE_SIZE = 224


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
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
    model.to(device)
    model.eval()
    return model, class_names


def load_image(path):
    image = Image.open(path)
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    else:
        image = image.convert('RGB')
    return image


def compute_gradcam(model, input_tensor, target_layer):
    """Υπολογίζει το Grad-CAM heatmap για την κλάση που προβλέπει το μοντέλο."""
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    try:
        output = model(input_tensor)
        predicted_idx = output.argmax(dim=1).item()
        confidence = F.softmax(output, dim=1)[0, predicted_idx].item()

        model.zero_grad()
        score = output[0, predicted_idx]
        score.backward()

        acts = activations[0][0]     # [C, H, W]
        grads = gradients[0][0]      # [C, H, W]

        weights = grads.mean(dim=(1, 2))  # [C] - global average pooling των gradients
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
        for c in range(acts.shape[0]):
            cam += weights[c] * acts[c]

        cam = F.relu(cam)  # μόνο θετική επιρροή μας ενδιαφέρει
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.detach().cpu().numpy(), predicted_idx, confidence
    finally:
        fh.remove()
        bh.remove()


def overlay_heatmap(original_image, cam, alpha=0.45):
    """Επικαλύπτει το heatmap πάνω στην αρχική εικόνα."""
    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        original_image.size, resample=Image.BILINEAR
    )
    cam_array = np.array(cam_resized) / 255.0

    colormap = cm.get_cmap("jet")
    heatmap_rgba = colormap(cam_array)  # [H, W, 4], τιμές 0-1
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_rgb)

    blended = Image.blend(original_image.convert("RGB"), heatmap_img, alpha=alpha)
    return blended


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM οπτικοποίηση για το fruit classifier")
    parser.add_argument("--image", required=True, help="Path στην εικόνα προς ανάλυση")
    parser.add_argument("--model", default="fruit_mobilenetv2.pth", help="Path στο μοντέλο (.pth)")
    parser.add_argument("--out", default=None, help="Path εξόδου (default: <όνομα_εικόνας>_gradcam.png)")
    args = parser.parse_args()

    device = torch.device("cpu")
    model, class_names = load_model(args.model, device)

    original_image = load_image(args.image)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    # Το τελευταίο conv block του MobileNetV2 features - εκεί "ζουν" τα πιο
    # υψηλού επιπέδου χωρικά χαρακτηριστικά (π.χ. σχήμα/υφή αντικειμένου)
    target_layer = model.features[-1]

    cam, predicted_idx, confidence = compute_gradcam(model, input_tensor, target_layer)
    predicted_class = class_names[predicted_idx]

    result_image = overlay_heatmap(original_image, cam)

    out_path = args.out or (os.path.splitext(args.image)[0] + "_gradcam.png")
    result_image.save(out_path)

    print(f"\nΠρόβλεψη: {predicted_class}  (confidence: {confidence*100:.2f}%)")
    print(f"Grad-CAM αποθηκεύτηκε στο: {out_path}")
    print("\nΚόκκινο/κίτρινο = εκεί εστιάζει το μοντέλο. Μπλε = αγνοεί.")
    print("Αν το κόκκινο πέφτει στο φόντο/άκρες αντί στο φρούτο, αυτό επιβεβαιώνει το domain gap.")


if __name__ == "__main__":
    main()