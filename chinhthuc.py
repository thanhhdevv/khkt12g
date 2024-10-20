from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import cv2
from ultralytics import YOLO
import threading
import google.generativeai as genai
from dotenv import load_dotenv


app = Flask(__name__)

# Đường dẫn đến thư mục lưu trữ
UPLOAD_FOLDER = 'uploads'
PROTEIN_FOLDER = 'protein'
old_image = None

# Tạo các thư mục nếu chưa tồn tại
os.makedirs(os.path.join(UPLOAD_FOLDER, 'plant_images'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'waste_images'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'fire_images'), exist_ok=True)
os.makedirs(PROTEIN_FOLDER, exist_ok=True)

model_leaf = YOLO('leaf.pt')
model_fire = YOLO('fire.pt')
model_waste = YOLO('waste.pt')

def setup_generative_model():
    """Load API key, configure the model, and start a chat session."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key is None:
        raise ValueError("API key not found. Please check your .env file.")
    
    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    
    return model.start_chat(history=[])

def send_message(chat_session, message):
    """Send a message to the chat session and return the response."""
    return chat_session.send_message(message).text

# Khởi động chat session và gửi tin nhắn
chat_session = setup_generative_model()
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    global old_image
    section = request.form.get('section')
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if old_image and os.path.exists(old_image):
        threading.Timer(600, os.remove, args=[old_image]).start()
    if section == 'plant_disease':
            img_path = os.path.join(UPLOAD_FOLDER, 'plant_images', file.filename)
            model = model_leaf
    elif section == 'waste_classification':
            img_path = os.path.join(UPLOAD_FOLDER, 'waste_images', file.filename)
            model = model_waste
    elif section == 'fire_detection':
            img_path = os.path.join(UPLOAD_FOLDER, 'fire_images', file.filename)
            model = model_fire
    else:
        return jsonify({'error': 'Invalid section'}), 400
    file.save(img_path)
    old_image = img_path 
    # Lưu file vào thư mục tương ứng
    try:
        image = cv2.imread(img_path)
        results = model.predict(image)
        max_conf = 0
        max_label = None

        for result in results:
              boxes = result.boxes
              for box in boxes:
                   conf = box.conf[0].item()
                   cls = int(box.cls[0].item())
                   label = model.names[cls]
                   if conf > max_conf:
                        max_conf = conf
                        max_label = label

        fire_label = ['fire','smoke','lighter']
        leaf_label = ['Bacterial Leaf Blight', 'Fungal Leaf Spot', 'Lack of Calcium', 'Leaf Spot', 'Yellow Leaf Curl Virus', 'Yellow Vein Mosaic Virus']
        vh_label = ['plato', 'circle-battery', 'square-battery'] 
        if max_label in fire_label and max_label in ['fire','smoke']:
             question = "hãy giúp tôi cách xử lí khi có cháy?"
        if max_label == "lighter":
             question = "cách xử lí vật dễ gây cháy nổ"
        if max_label in leaf_label:
             question = max_label + " là bệnh gì?Cách xử lí bệnh này thế nào?Cách phòng chống,biện pháp phòng chống lâu dài"
        if max_label in vh_label:
            if max_label == "plato":

                max_label = "thucan"

            question = max_label + " là vật vô cơ hay hữu cơ.Cách xử lí để bảo vệ môi trường"
        response_text = send_message(chat_session, question)
        print(response_text)
        print(f'Max Label: {max_label}, Max Confidence: {max_conf}')

        return jsonify({'label': max_label, 'confidence': max_conf, 'resp': response_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'frame' not in request.files:
        return jsonify({'error': 'Không có video'}), 400

    frame = request.files['frame'].read()
    np_image = np.frombuffer(frame, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    question = None
    # Dự đoán chỉ khi có hình ảnh
    if image is not None:
        results = model_fire.predict(image)
        max_conf = 0
        max_label = None
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = model_fire.names[cls]
                if conf > max_conf:
                    max_conf = conf
                    max_label = label
        if max_label in ['fire','smoke']:
             question = "hãy giúp tôi cách xử lí khi có cháy?"
        if max_label == "lighter":
             question = "cách xử lí vật dễ gây cháy nổ"
        
        if question is not None:
            response_text = send_message(chat_session, question)
            print(response_text)

        if max_label is not None:
            print(f'Max Label: {max_label}, Max Confidence: {max_conf}')
            return jsonify({'label': max_label, 'confidence': max_conf, 'resp1':response_text})
    return jsonify({'error': 'Không có đối tượng nào được phát hiện.'}), 500


if __name__ == '__main__':
    app.run(port=5000,debug=True)
