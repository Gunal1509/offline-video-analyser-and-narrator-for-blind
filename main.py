import os
import cv2
import ffmpeg
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pyttsx3
from tkinter import Tk, filedialog, messagebox

# ----- Frame + Audio Extractor -----
def extract_frames(video_path, interval=2, output_dir="frames"):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % int(fps * interval) == 0:
            img_name = os.path.join(output_dir, f"frame_{i}.jpg")
            cv2.imwrite(img_name, frame)
            frames.append(img_name)
        i += 1
    cap.release()
    print(f"[✔] Extracted {len(frames)} frames to '{output_dir}'")
    return frames

def extract_audio(video_path, audio_out="whisper.cpp/audio.wav"):
    os.makedirs(os.path.dirname(audio_out), exist_ok=True)
    try:
        ffmpeg.input(video_path).output(audio_out, acodec='pcm_s16le', ac=1, ar='16000').run(overwrite_output=True, quiet=True)
        print(f"[✔] Audio extracted to '{audio_out}'")
    except ffmpeg.Error as e:
        print(f"[❌] Error extracting audio:\n{e.stderr.decode()}")

# ----- Caption + TTS -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def describe_image(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.strip().rstrip(".")

def process_all_frames(frames_dir="frames", output_file="descriptions.txt"):
    unique_descriptions = set()
    image_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    for idx, file_name in enumerate(image_files):
        full_path = os.path.join(frames_dir, file_name)
        print(f"[{idx+1}/{len(image_files)}] Processing {file_name}...")
        try:
            description = describe_image(full_path)
            unique_descriptions.add(description)
        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")
    paragraph = '. '.join(sorted(unique_descriptions)) + '.'
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(paragraph)
    print(f"\n✅ Descriptions saved in: {output_file}")
    return paragraph

def narrate_paragraph(text):
    print("\n🔊 Narrating paragraph...")
    engine.say(text)
    engine.runAndWait()

# ----- GUI -----
def browse_and_process():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select MP4 File",
        filetypes=[("MP4 files", "*.mp4")]
    )
    if not file_path:
        messagebox.showinfo("Cancelled", "No file selected.")
        return
    print(f"🎞 Selected file: {file_path}")

    extract_frames(file_path, interval=2, output_dir="frames")
    extract_audio(file_path, audio_out="whisper.cpp/audio.wav")
    paragraph = process_all_frames()
    narrate_paragraph(paragraph)
    messagebox.showinfo("Done", "Narration complete!")

if __name__ == "__main__":
    browse_and_process()
