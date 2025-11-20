import gradio as gr
import cv2
import whisper
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- 1. SETUP & LOAD MODELS ---
print("Sedang memuat model... Mohon tunggu sebentar.")

# A. Model Otak: SmolLM (Agent)
# Menggunakan versi Instruct agar bisa diajak diskusi
model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
smol_lm = AutoModelForCausalLM.from_pretrained(model_id)

# B. Model Telinga: Whisper (Audio to Text)
whisper_model = whisper.load_model("tiny")

# C. Model Mata: Vision Transformer untuk Emosi
# Kita ganti FER dengan model native Hugging Face agar tidak error
emotion_classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

# D. Setup Deteksi Wajah (OpenCV Basic)
# Menggunakan Haar Cascade bawaan cv2 untuk menemukan lokasi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 2. FUNGSI LOGIKA ---

def get_dominant_emotion(video_path):
    cap = cv2.VideoCapture(video_path)
    emotions_list = []
    frame_count = 0
    
    # Ambil sampel setiap 30 frame (sekitar 1 detik sekali)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 30 == 0:
            # 1. Convert ke Grayscale untuk deteksi wajah
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # 2. Crop bagian wajah saja
                face_roi = frame[y:y+h, x:x+w]
                
                # 3. Convert ke format PIL Image untuk Hugging Face Pipeline
                rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_face)
                
                # 4. Prediksi Emosi
                try:
                    results = emotion_classifier(pil_image)
                    # results format: [{'label': 'happy', 'score': 0.9}, ...]
                    top_emotion = results[0]['label']
                    emotions_list.append(top_emotion)
                except Exception as e:
                    print(f"Error detecting frame: {e}")
                    continue
                
                # Kita hanya ambil 1 wajah pertama yang ketemu per frame
                break 
                
        frame_count += 1
    cap.release()
    
    if not emotions_list:
        return "Tidak ada wajah terdeteksi"
    
    # Cari modus (emosi yang paling sering muncul)
    return max(set(emotions_list), key=emotions_list.count)

def analyze_agent(video_path):
    if not video_path:
        return "Error", "Mohon upload video.", "N/A"

    print(f"Processing video: {video_path}")

    # 1. Transkripsi Audio (Telinga)
    try:
        audio_result = whisper_model.transcribe(video_path)
        transcribed_text = audio_result["text"]
    except Exception as e:
        transcribed_text = f"Gagal transkripsi audio: {str(e)}"

    # 2. Deteksi Emosi Visual (Mata)
    detected_emotion = get_dominant_emotion(video_path)
    
    # 3. Analisis SmolLM (Otak)
    system_prompt = "You are an expert AI psychological analyst. Analyze the user's emotion based on facial expression and text."
    
    user_input = f"""
    DATA DARI USER:
    - Teks Ucapan: "{transcribed_text}"
    - Ekspresi Wajah Dominan: {detected_emotion}
    
    INSTRUKSI:
    Analisis apakah ada kesesuaian antara ucapan dan ekspresi wajahnya. 
    Jika wajah 'sad' tapi teks semangat, mungkin dia menyembunyikan sesuatu.
    Berikan kesimpulan singkat dalam Bahasa Indonesia.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    
    # Format chat template
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=True)
    
    # Generate response
    outputs = smol_lm.generate(input_ids, max_new_tokens=250, temperature=0.7)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parsing output agar rapi (mengambil bagian assistant saja)
    if "assistant" in decoded:
        final_response = decoded.split("assistant")[-1].strip()
    else:
        # Fallback jika format berbeda
        final_response = decoded

    return final_response, transcribed_text, detected_emotion

# --- 3. USER INTERFACE ---

css = """
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("## 🤖 SmolLM3 Multimodal Agent (Video Emotion)")
        gr.Markdown("Upload video Anda berbicara. AI akan melihat ekspresi wajah dan mendengar ucapan Anda.")
        
        video_input = gr.Video(sources=["upload", "webcam"])
        submit_btn = gr.Button("Analisis Emosi", variant="primary")
        
        gr.Markdown("### Hasil Analisis Agent")
        output_agent = gr.Textbox(label="Pendapat SmolLM3", lines=4)
        
        with gr.Row():
            output_text = gr.Textbox(label="Transkrip Suara")
            output_face = gr.Textbox(label="Deteksi Wajah")

    submit_btn.click(
        fn=analyze_agent,
        inputs=[video_input],
        outputs=[output_agent, output_text, output_face]
    )

if __name__ == "__main__":
    demo.launch()