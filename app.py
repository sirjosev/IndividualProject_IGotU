import gradio as gr
import cv2
import whisper
# --- 2. FUNGSI LOGIKA ---

def get_emotion_analysis(video_path):
    """
    Menganalisis video dan mengembalikan distribusi emosi dalam bentuk persentase.
    """
    cap = cv2.VideoCapture(video_path)
    emotions_list = []
    frame_count = 0
    
    # Ambil sampel setiap 15 frame (sekitar 0.5 detik sekali) untuk data lebih rapat
    sample_rate = 15 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            # 1. Convert ke Grayscale untuk deteksi wajah
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # 2. Crop bagian wajah saja
                face_roi = frame[y:y+h, x:x+w]
                
                # 3. Convert ke format PIL Image
                rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_face)
                
                # 4. Prediksi Emosi
                try:
                    results = emotion_classifier(pil_image)
                    # results format: [{'label': 'happy', 'score': 0.9}, ...]
                    top_emotion = results[0]['label']
                    emotions_list.append(top_emotion)
                except Exception as e:
                    pass # Skip frame error
                
                # Ambil 1 wajah utama saja
                break 
                
        frame_count += 1
    cap.release()
    
    if not emotions_list:
        return "Tidak ada wajah terdeteksi", "N/A"
    
    # Hitung Persentase
    total_frames = len(emotions_list)
    counts = Counter(emotions_list)
    
    # Format string deskriptif
    analysis_str = []
    for emotion, count in counts.most_common():
        percent = (count / total_frames) * 100
        analysis_str.append(f"{emotion} ({percent:.1f}%)")
    
    dominant_emotion = counts.most_common(1)[0][0]
    detailed_analysis = ", ".join(analysis_str)
    
    return dominant_emotion, detailed_analysis

def analyze_agent(video_path):
    if not video_path:
        return "Error: Mohon upload video.", "N/A", "N/A"

    print(f"Processing video: {video_path}")

    # 1. Transkripsi Audio (Telinga)
    try:
        audio_result = whisper_model.transcribe(video_path)
        transcribed_text = audio_result["text"]
    except Exception as e:
        transcribed_text = f"[Error Transkripsi]: {str(e)}"

    # 2. Deteksi Emosi Visual (Mata)
    dom_emotion, detailed_emotion = get_emotion_analysis(video_path)
    
    # 3. Analisis Otak (LLM)
    system_prompt = (
        "Anda adalah ahli psikologi perilaku dan pembaca mikro-ekspresi (micro-expressions) yang sangat teliti. "
        "Tugas Anda adalah menganalisis kejujuran dan kondisi emosional seseorang berdasarkan data visual dan verbal.\n"
        "FOKUS UTAMA: Perhatikan ketidaksesuaian antara ucapan dan 'sorot mata' (emosi wajah yang terdeteksi).\n"
        "Jawab dalam Bahasa Indonesia yang profesional namun tajam."
    )
    
    user_input = f"""
    DATA OBSERVASI:
    1. TRANSKRIP UCAPAN: "{transcribed_text}"
    
    2. ANALISIS WAJAH (MATA & EKSPRESI):
    - Emosi Dominan: {dom_emotion}
    - Detail Distribusi Emosi: {detailed_emotion}
    
    TUGAS ANDA:
    Analisis apakah orang ini jujur atau menyembunyikan sesuatu. 
    - Jika distribusi emosi bercampur (misal: Dominan Happy tapi ada Fear/Sad), curigai itu sebagai "smiling depression" atau kecemasan tersembunyi.
    - Hubungkan konteks ucapan dengan emosi yang muncul.
    - Berikan kesimpulan psikologis singkat.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    
    # Format chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(llm_model.device)
    
    # Generate response
    generated_ids = llm_model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        temperature=0.7
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response, transcribed_text, detailed_emotion

# --- 3. USER INTERFACE ---

css = """
#col-container {max-width: 800px; margin-left: auto; margin-right: auto;}
.feedback-box {border: 1px solid #ddd; padding: 10px; border-radius: 8px; background-color: #f9f9f9;}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("## 👁️ IGotU - AI Micro-Expression Analyst")
        gr.Markdown(
            "Upload video pernyataan seseorang. Agent ini akan membaca **sorot mata dan mikro-ekspresi** "
            "mereka untuk mendeteksi emosi tersembunyi dibalik ucapan."
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(sources=["upload", "webcam"], label="Input Video")
                submit_btn = gr.Button("🔍 Analisis Kebenaran", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 🧠 Analisis Psikologis")
                output_agent = gr.Textbox(
                    label="Laporan Agent", 
                    lines=10, 
                    placeholder="Hasil analisis akan muncul di sini...",
                    show_copy_button=True
                )
        
        with gr.Accordion("Lihat Data Mentah (Transkrip & Deteksi)", open=False):
            with gr.Row():
                output_text = gr.Textbox(label="Transkrip Audio (Whisper Base)")
                output_face = gr.Textbox(label="Distribusi Emosi (Vision AI)")

    submit_btn.click(
        fn=analyze_agent,
        inputs=[video_input],
        outputs=[output_agent, output_text, output_face]
    )

if __name__ == "__main__":
    demo.launch()