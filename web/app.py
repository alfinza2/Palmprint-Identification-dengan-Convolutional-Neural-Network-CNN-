import os
import base64
import math
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import mediapipe as mp

app = Flask(__name__)

# Dictionary for class labels
dic = {
    0: '001',
    1: '002',
    2: '003',
    3: '004',
    4: '005',
    5: '006',
    6: '007',
    7: '008',
    8: '009',
    9: '010',
    10: '011',
    11: '012',
    12: '013',
    13: '014',
    14: '015',
    15: '016',
    16: '017',
    17: '018',
    18: '019',
    19: '020',
    20: '021',
    21: '022',
    22: '023',
    23: '024',
    24: '025',
    25: '026',
    26: '027',
    27: '028',
    28: '029',
    29: '030',
    30: '031',
    31: '032',
    32: '033',
    33: '034',
    34: '035',
    35: '036',
    36: '037',
    37: '038',
    38: '039',
    39: '040',
    40: '041',
    41: 'Alfin',
    42: '043',
    43: '044',
    44: '045',
    45: '046',
    46: '047',
    47: '048',
    48: '049',
    49: '050'
}

# Load the trained model
model = load_model('PalmPrint-Identification-96.44.h5')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.2)

def resize_image(image, size=(224, 224)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

# Function to calculate LBP
def calculate_lbp(image):
    height, width = image.shape
    lbp_image = np.zeros((height, width), dtype=np.uint8)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    
    for y in range(1, height-1):
        for x in range(1, width-1):
            center = image[y, x]
            binary_string = ''
            
            for dy, dx in neighbors:
                neighbor_value = image[y + dy, x + dx]
                binary_string += '1' if neighbor_value >= center else '0'
            
            lbp_value = int(binary_string, 2)
            lbp_image[y, x] = lbp_value
    
    return lbp_image

# Function to rotate image based on angle
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(image, M, (new_w, new_h))
    return rotated

# Function to get the rotation angle to make the palm facing up
def get_rotation_angle(landmarks):
    x1, y1 = landmarks[0].x, landmarks[0].y
    x2, y2 = landmarks[9].x, landmarks[9].y
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle + 90

# Function to crop the hand region from the image
def crop_hand(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = image.shape
            selected_landmarks = [0, 1, 5, 9, 13, 17]
            x_min = int(min([hand_landmarks.landmark[i].x for i in selected_landmarks]) * w)
            y_min = int(min([hand_landmarks.landmark[i].y for i in selected_landmarks]) * h)
            x_max = int(max([hand_landmarks.landmark[i].x for i in selected_landmarks]) * w)
            y_max = int(max([hand_landmarks.landmark[i].y for i in selected_landmarks]) * h)
            margin = 10
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)
            cropped_image = image[y_min:y_max, x_min:x_max]
            gray_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            blurred_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
            return blurred_img

# Function to process LBP for images in a folder
def process_lbp_images(input_dir, output_lbp_dir):
    if not os.path.exists(output_lbp_dir):
        os.makedirs(output_lbp_dir)

    for id_folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, id_folder)
        output_folder_path = os.path.join(output_lbp_dir, id_folder)

        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        for filename in os.listdir(folder_path):
            if filename.endswith(('.JPG', '.jpg')):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                lbp_image = calculate_lbp(image)
                output_path = os.path.join(output_folder_path, filename)
                cv2.imwrite(output_path, lbp_image)

# Route for the main page
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("classification.html")

# Route for handling image uploads
@app.route("/submit", methods=['POST'])
def get_hours():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = os.path.join('static', img.filename)
        img.save(img_path)

        # Preprocessing the image
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            rotation_angle = get_rotation_angle(landmarks)
            rotated_image = rotate_image(image, rotation_angle)
            cropped_image = crop_hand(rotated_image)
            
            # Save rotated and cropped images
            rotate_dir = os.path.join('static', 'rotate')
            crop_dir = os.path.join('static', 'crop')
            lbp_dir = os.path.join('static', 'lbp')
            if not os.path.exists(rotate_dir):
                os.makedirs(rotate_dir)
            if not os.path.exists(crop_dir):
                os.makedirs(crop_dir)
            if not os.path.exists(lbp_dir):
                os.makedirs(lbp_dir)
            
            rotated_path = os.path.join(rotate_dir, img.filename)
            cropped_path = os.path.join(crop_dir, img.filename)
            lbp_path = os.path.join(lbp_dir, img.filename)
            
            cv2.imwrite(rotated_path, rotated_image)
            cv2.imwrite(cropped_path, cropped_image)
            
            # Perform LBP calculation
            lbp_image = calculate_lbp(cropped_image)
            cv2.imwrite(lbp_path, lbp_image)

            # Perform prediction on LBP image
            cv2.imwrite(img_path, lbp_image)
            prediction_id = predict_label(img_path)

            # Set path to the thumbnail image
            thumbnail_path = os.path.join('static/thumbnails', f'{prediction_id}.png')

            return render_template("classification.html", prediction=prediction_id, img_path=thumbnail_path)
    return render_template("classification.html")

# Route for handling webcam image submissions
@app.route("/submit_webcam", methods=['POST'])
def submit_webcam():
    if request.method == 'POST':
        img_data = request.form['webcam_image']
        img_data = img_data.replace('data:image/png;base64,', '')
        img_data = base64.b64decode(img_data)
        img_path = os.path.join('static', 'webcam_image.png')
        with open(img_path, "wb") as f:
            f.write(img_data)

        # Preprocessing the webcam image
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            rotation_angle = get_rotation_angle(landmarks)
            rotated_image = rotate_image(image, rotation_angle)
            cropped_image = crop_hand(rotated_image)
            
            # Save rotated and cropped images
            rotate_dir = os.path.join('static', 'rotate')
            crop_dir = os.path.join('static', 'crop')
            lbp_dir = os.path.join('static', 'lbp')
            if not os.path.exists(rotate_dir):
                os.makedirs(rotate_dir)
            if not os.path.exists(crop_dir):
                os.makedirs(crop_dir)
            if not os.path.exists(lbp_dir):
                os.makedirs(lbp_dir)
            
            rotated_path = os.path.join(rotate_dir, 'webcam_image.png')
            cropped_path = os.path.join(crop_dir, 'webcam_image.png')
            lbp_path = os.path.join(lbp_dir, 'webcam_image.png')
            
            cv2.imwrite(rotated_path, rotated_image)
            cv2.imwrite(cropped_path, cropped_image)
            
            # Perform LBP calculation
            lbp_image = calculate_lbp(cropped_image)
            cv2.imwrite(lbp_path, lbp_image)

            # Perform prediction on LBP image
            cv2.imwrite(img_path, lbp_image)
            prediction_id = predict_label(img_path)

            # Set path to the thumbnail image
            thumbnail_path = os.path.join('static/thumbnails', f'{prediction_id}.png')

            return render_template("classification.html", prediction=prediction_id, img_path=thumbnail_path)
    return render_template("classification.html")

# Function to predict label based on the image
def predict_label(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    p = model.predict(img)
    return dic[np.argmax(p)]

if __name__ == '__main__':
    app.run(debug=True)

# Clean up
hands.close()
