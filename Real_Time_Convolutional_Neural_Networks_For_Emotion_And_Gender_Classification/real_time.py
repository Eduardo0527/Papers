import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from Xception import MiniXception

# 1. Setup Data
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
model_weights = 'mini_xception_best.pth'

# 2. Initialize Model ONCE before the webcam loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MiniXception(in_channels=1, num_classes=len(classes))

if not os.path.exists(model_weights):
    print(f"Error: Could not find model weights at '{model_weights}'.")
    exit()

model.load_state_dict(torch.load(model_weights, map_location=device))
model.to(device)
model.eval() 

# 3. Setup Image Transforms
transform = transforms.Compose([
    transforms.Resize((48, 48)), 
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict_emotion(image_array):
    """Takes a cv2 image array, converts it, and predicts the emotion."""
    # Convert OpenCV BGR array to RGB, then to PIL Image
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Apply transforms and add batch dimension
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad(): 
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        top_prob, top_class = torch.max(probabilities, 1)

    predicted_emotion = classes[top_class.item()]
    confidence = top_prob.item() * 100

    return predicted_emotion.upper(), confidence

if __name__ == "__main__":
    # Added cv2.CAP_DSHOW to bypass the MSMF Windows error
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Error: Could not open webcam. Check Windows privacy settings or other apps.")
        exit()

    print("Press 'q' to quit.")
    
    try:
        while True:
            success, image = cap.read()
            
            if not success:
                print("Failed to capture frame")
                break
            
            # Flip image like a mirror for easier viewing
            image = cv2.flip(image, 1)
            
            # Get Prediction
            emotion, conf = predict_emotion(image)
            
            # Draw the text on the webcam feed
            text = f"{emotion} ({conf:.1f}%)"
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Show the frame
            cv2.imshow("Real-Time Emotion Recognition", image)
            
            # Quit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()