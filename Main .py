import cv2
import time
import requests
import RPi.GPIO as GPIO
from ultralytics import YOLO

# Load the pre-trained YOLO model
model = YOLO('yolov8n.pt')

# Open webcam (usually index 0 or 1)
cap = cv2.VideoCapture(0)

# Buzzer setup
BUZZER_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# Get video details
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# Define VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_webcam.avi', fourcc, fps, (width, height))

# Telegram bot config
TELEGRAM_BOT_TOKEN = '8122770632:AAECklfwIoW7ePUBwN2Db8eT0rYxc2N8Rzo'
CHAT_ID = '328102547'

def send_telegram_message(message, image_path=None):
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    requests.get(url, params={'chat_id': CHAT_ID, 'text': message})

    if image_path:
        url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto'
        with open(image_path, 'rb') as image_file:
            requests.post(url, files={'photo': image_file}, params={'chat_id': CHAT_ID})

def interpolate_box(last_box, next_box, alpha):
    x1 = int(last_box[0] * (1 - alpha) + next_box[0] * alpha)
    y1 = int(last_box[1] * (1 - alpha) + next_box[1] * alpha)
    x2 = int(last_box[2] * (1 - alpha) + next_box[2] * alpha)
    y2 = int(last_box[3] * (1 - alpha) + next_box[3] * alpha)
    return (x1, y1, x2, y2)

# Tracking state
person_tracking = {}
no_movement_duration = 4
last_telegram_sent_time = time.time()
message_interval = 30
frame_count = 0
drowning_detected = False
image_path = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    current_time = time.time()
    detected_person_ids = set()
    new_drowning_person_detected = False

    for result in results.boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
        conf = result.conf[0].item()
        cls = int(result.cls[0].item())
        label = model.names[cls]

        if label == 'person':
            person_id = f"{x1}_{y1}_{x2}_{y2}"
            detected_person_ids.add(person_id)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if person_id not in person_tracking:
                person_tracking[person_id] = {
                    'last_box': (x1, y1, x2, y2),
                    'last_movement_time': current_time,
                    'last_frame': frame_count,
                    'next_box': (x1, y1, x2, y2),
                    'next_frame': frame_count,
                    'drowning_flagged': False
                }
            else:
                last_box = person_tracking[person_id]['last_box']
                distance_moved = ((center_x - (last_box[0] + last_box[2]) // 2) ** 2 +
                                  (center_y - (last_box[1] + last_box[3]) // 2) ** 2) ** 0.5

                if distance_moved < 10:
                    time_no_movement = current_time - person_tracking[person_id]['last_movement_time']
                    if time_no_movement > no_movement_duration and not person_tracking[person_id]['drowning_flagged']:
                        person_tracking[person_id]['drowning_flagged'] = True
                        new_drowning_person_detected = True
                        drowning_detected = True
                        print(f"Drowning detected for person {person_id} at frame {frame_count}")

                person_tracking[person_id]['last_movement_time'] = current_time
                person_tracking[person_id]['last_box'] = (x1, y1, x2, y2)
                person_tracking[person_id]['last_frame'] = frame_count

            color = (0, 0, 255) if person_tracking[person_id]['drowning_flagged'] else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            person_tracking[person_id]['next_box'] = (x1, y1, x2, y2)
            person_tracking[person_id]['next_frame'] = frame_count

    if new_drowning_person_detected and (current_time - last_telegram_sent_time >= message_interval):
        image_path = f"drowning_{frame_count}.jpg"
        cv2.imwrite(image_path, frame)
        send_telegram_message(f"🚨 EMERGENCY: Drowning detected at frame {frame_count}.", image_path)
        last_telegram_sent_time = current_time
        break  # 💥 Exit loop immediately

    for person_id, data in person_tracking.items():
        if person_id not in detected_person_ids and frame_count - data['last_frame'] <= 10:
            alpha = (frame_count - data['last_frame']) / (data['next_frame'] - data['last_frame'] + 1e-5)
            interpolated_box = interpolate_box(data['last_box'], data['next_box'], alpha)
            x1, y1, x2, y2 = interpolated_box
            color = (0, 0, 255) if drowning_detected else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, 'Interpolated', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if drowning_detected:
        text = "🚨 EMERGENCY: DROWNING"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
        text_x = (width - text_size[0]) // 2
        text_y = 60
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    out.write(frame)
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Clean-up
cap.release()
out.release()
cv2.destroyAllWindows()

if drowning_detected:
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(10)  # ✅ Beep for 1 minute
    GPIO.output(BUZZER_PIN, GPIO.LOW)

GPIO.cleanup()
