import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
from twilio.rest import Client
import os
import time
from datetime import datetime

# ================= CONFIGURATION =================
YOLO_MODEL_PATH = "./best.pt"
CNN_MODEL_PATH = "./cnn_severity_model.pth"

# Twilio SMS configuration
account_sid = "ACe873aa5742fda189b13eac1f8850c4b3"
auth_token = "3b9b9b3e5199c6a515cf508ddbc2c7ed"
twilio_number = "+16165121598"

recipients = [
    "+917893046524",
    "+919515974503",
    "+919003395395"
]

ACCIDENT_THRESHOLD = 0.30
FRAME_SKIP = 3

# ================= INITIALIZE MODELS =================
@st.cache_resource
def load_models():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    yolo = YOLO(YOLO_MODEL_PATH).to(device)

    cnn = models.mobilenet_v3_small(weights=None)
    cnn.classifier[3] = nn.Linear(1024, 2)
    cnn.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    cnn = cnn.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return yolo, cnn, transform, device

# ================= SMS FUNCTION =================
def send_bulk_sms(numbers, message_body):
    client = Client(account_sid, auth_token)

    for number in numbers:
        try:
            message = client.messages.create(
                body=message_body,
                from_=twilio_number,
                to=number
            )
            print(f"✅ Sent to {number} | SID: {message.sid}")
        except Exception as e:
            print(f"❌ Failed for {number} | Error: {e}")

# ================= FRAME PROCESSING =================
def process_frame(frame, yolo, cnn, transform, device):
    display_frame = frame.copy()
    results = yolo(frame, conf=0.35, verbose=False)[0]
    max_prob = 0.0

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensor = transform(crop_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = cnn(tensor)
                prob = torch.softmax(output, dim=1)[0][1].item()
                max_prob = max(max_prob, prob)

            color = (0, 0, 255) if prob >= ACCIDENT_THRESHOLD else (0, 255, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

    return display_frame, max_prob

# ================= STREAMLIT APP =================
def main():
    st.set_page_config(page_title="Accident Detection", page_icon="🚗")
    st.title("🚗 AI Accident Detection System")
    st.markdown("---")

    yolo, cnn, transform, device = load_models()

    st.sidebar.header("⚙️ Settings")
    global ACCIDENT_THRESHOLD
    ACCIDENT_THRESHOLD = st.sidebar.slider(
        "Accident Threshold",
        0.0, 1.0, 0.30, 0.05
    )

    st.subheader("Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'mov', 'avi', 'mkv']
    )

    if uploaded_file and st.button("🚀 Start Detection"):

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        frame_probs = []
        high_prob_frames = 0
        frame_count = 0

        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % FRAME_SKIP != 0:
                continue

            processed_frame, prob = process_frame(
                frame, yolo, cnn, transform, device
            )

            frame_probs.append(prob)

            if prob >= ACCIDENT_THRESHOLD:
                high_prob_frames += 1

            display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_frame)

        cap.release()

        st.markdown("---")
        st.subheader("📊 Final Results")

        st.write(f"Total Frames Processed: {len(frame_probs)}")
        st.write(f"High Probability Frames: {high_prob_frames}")

        # ================= NEW DETECTION RULE =================
        if high_prob_frames > 5:
            st.error("🚨 ACCIDENT DETECTED!")

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            message_body = f"""
🚨 ACCIDENT ALERT 🚨

High probability frames: {high_prob_frames}
Camera 1
Time: {current_time}

Immediate attention required.
"""

            send_bulk_sms(recipients, message_body)

            st.success("📩 SMS alerts sent to emergency contacts.")

        else:
            st.success("✅ No Accident Detected")

if __name__ == "__main__":
    main()
