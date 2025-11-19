import gradio as gr
import cv2
import whisper
import torch
from fer import FER
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- 1. LOAD MODELS ---
print("Sedang memuat model... Mohon tunggu.")

# A. Model Otak: SmolLM (Gunakan versi Instruct agar bisa diajak chat/analisis)
# Kita gunakan versi 135M atau 360M agar ringan di CPU Hugging Face Space gratis
model_id = "HuggingFaceTB/SmolLM-135M-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_id)
smol_lm = AutoModelForCausalLM.from_pretrained(model_id)

# B. Model Telinga: Whisper untuk Transkripsi
whisper_model = whisper.load_model("tiny") # Gunakan 'tiny' atau 'base' agar cepat

# C. Model Mata: FER untuk Emosi Wajah
face_detector = FER(mtcnn=True) # MTCNN lebih akurat

# --- 2. FUNGSI PEMROSESAN ---

def analyze_emotion(video_path):
    """
    Fungsi utama yang memproses video user.
    """
    if not video_path:
        return "Mohon upload video terlebih dahulu."

    # --- LANGKAH 1: Analisis Audio (Transkripsi) ---
    # Whisper otomatis ekstrak audio dari file video
    audio_result = whisper_model.transcribe(video_path)
    transcribed_text = audio_result["text"]
    
    # --- LANGKAH 2: Analisis Visual (Mimik Muka) ---
    # Kita ambil beberapa frame dari video untuk dicek emosinya
    cap = cv2.VideoCapture(video_path)
    emotions_list = []
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Cek setiap 30 frame (agar tidak terlalu berat)
        if frame_count % 30 == 0:
            # FER mendeteksi emosi dominan di frame ini
            top_emotion, score = face_detector.top_emotion(frame)
            if top_emotion:
                emotions_list.append(top_emotion)
        frame_count += 1
    cap.release()

    # Hitung emosi yang paling sering muncul (Modus)
    if emotions_list:
        dominant_facial_emotion = max(set(emotions_list), key=emotions_list.count)
    else:
        dominant_facial_emotion = "Netral/Tidak Terdeteksi"

    # --- LANGKAH 3: Analisis Agen (SmolLM3) ---
    # Kita buat prompt agar SmolLM bertindak sebagai psikolog/analis
    
    system_prompt = "You are an expert AI emotional analyst. Analyze the user's state based on their facial expression and spoken words."
    
    user_input = f"""
    DATA INPUT:
    1. Transcribed Text: "{transcribed_text}"
    2. Facial Expression Detected: {dominant_facial_emotion}
    
    TUGAS:
    Jelaskan emosi apa yang dirasakan orang ini? Apakah kata-katanya (teks) cocok dengan ekspresi wajahnya? Berikan kesimpulan singkat dalam Bahasa Indonesia.
    """
    
    # Format prompt sesuai template chat SmolLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    
    # Generate jawaban
    outputs = smol_lm.generate(inputs, max_new_tokens=200, temperature=0.7, top_p=0.9)
    analysis_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Bersihkan output (hapus prompt asli dari hasil)
    final_response = analysis_result.split("assistant")[-1].strip()

    return final_response, transcribed_text, dominant_facial_emotion

# --- 3. MEMBUAT UI DENGAN GRADIO ---

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 SmolLM3 Emotion Agent")
    gr.Markdown("Upload video pendek (berbicara ke kamera), AI akan mendeteksi: **Mimik Wajah + Teks Ucapan**.")
    
    with gr.Row():
        video_input = gr.Video(label="Upload Video atau Rekam via Webcam")
        
    submit_btn = gr.Button("Analisis Emosi")
    
    with gr.Row():
        output_analysis = gr.Textbox(label="Analisis SmolLM3 (Agent)", lines=5)
    
    with gr.Row():
        output_text = gr.Textbox(label="Teks Terdeteksi")
        output_face = gr.Textbox(label="Emosi Wajah Dominan")

    submit_btn.click(
        fn=analyze_emotion, 
        inputs=video_input, 
        outputs=[output_analysis, output_text, output_face]
    )

# Jalankan aplikasi
demo.launch()