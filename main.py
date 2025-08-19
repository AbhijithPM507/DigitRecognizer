import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# Load your model
from model import Net
model = Net()
model.load_state_dict(torch.load(
    'model/digit_model.pth',
    map_location=torch.device('cpu'),
    weights_only=True  # safer loading
))
model.eval()

# Try to detect if OpenCV GUI is available
cv_gui_available = hasattr(cv2, "namedWindow")

# Create a blank canvas
canvas = np.ones((280, 280), dtype=np.uint8) * 255
drawing = False

def draw(event, x, y, flags, param):
    global drawing, canvas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), 8, (0,), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def predict_digit(img):
    img_resized = cv2.resize(img, (28, 28))
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        output = model(img_tensor)
        return output.argmax(dim=1).item()

if cv_gui_available:
    # --- OpenCV display mode ---
    cv2.namedWindow('Draw Digit')
    cv2.setMouseCallback('Draw Digit', draw)

    while True:
        cv2.imshow('Draw Digit', canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('p'):  # predict
            pred = predict_digit(canvas)
            print(f"Predicted Digit: {pred}")
        elif key == ord('c'):  # clear
            canvas[:] = 255
        elif key == 27:  # ESC to quit
            break

    cv2.destroyAllWindows()

else:
    # --- Matplotlib fallback ---
    print("⚠ OpenCV GUI not available — using matplotlib for display.")
    print("Draw in the image array directly or load an image file.")

    # Example: just show the blank canvas
    plt.imshow(canvas, cmap='gray')
    plt.axis('off')
    plt.show()

    # You can still run prediction if you have an image
    pred = predict_digit(canvas)
    print(f"Predicted Digit: {pred}")
