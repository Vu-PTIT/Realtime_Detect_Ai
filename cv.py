from ultralytics import YOLO
import cv2
import tensorflow as tf
import numpy as np

# Load YOLO model
yolo_model = YOLO('yolo11n.pt')  # hoặc 'yolo11n.pt' nếu bạn có

# Load Keras model (.h5)
keras_model = tf.keras.models.load_model('violence_detection_model.h5')


# Mở video
cap = cv2.VideoCapture('demo.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện bằng YOLO
    results = yolo_model(frame)

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # class 'person'
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Crop vùng người
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue  # Tránh lỗi khi crop trống

                # Resize để phù hợp model
                person_resized = cv2.resize(person_crop, (128, 128))  # Sửa theo input của model bạn
                person_resized = person_resized / 255.0
                person_input = np.expand_dims(person_resized, axis=0)

                # Dự đoán
                prediction = keras_model.predict(person_input, verbose=0)[0]
                label_idx = np.argmax(prediction)
                confidence = prediction[label_idx]
                label_name = 'Violent' if confidence > 0.5 else 'Non_violence'  # Cập nhật theo mô hình bạn

                # Text hiển thị
                text = f'{label_name} ({confidence:.2f})'

                # Vẽ khung và nhãn
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
