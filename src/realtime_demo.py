import argparse
from collections import deque
import cv2
import torch
from torchvision import transforms
from model import ASLClassifierCNN

CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]


def main(weights_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ASLClassifierCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(0)
    prediction_buffer = deque(maxlen=10)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        # Define Region of Interest (ROI) for hand sign
        x1, y1, x2, y2 = 100, 100, 350, 350
        roi = frame[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        input_tensor = transform(roi_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        if conf.item() > 0.75:
            prediction_buffer.append(CLASSES[pred_idx.item()])

        stable_pred = (
            max(set(prediction_buffer), key=prediction_buffer.count)
            if prediction_buffer else "..."
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Sign: {stable_pred} ({conf.item()*100:.1f}%)",
            (x1, y1 - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("ASL Alphabet Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="checkpoints/best_asl_cnn.pth")
    args = parser.parse_args()
    main(args.weights)
