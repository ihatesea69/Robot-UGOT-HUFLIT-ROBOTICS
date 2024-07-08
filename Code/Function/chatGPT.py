from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import torch
import requests

# - Cài đặt thư viện cần thiết:
# - pip install transformers librosa torch
# - pip install requests

# Tải mô hình nhận dạng giọng nói:
# Sử dụng mô hình khanhld/wav2vec2-base-vietnamese-160h từ Hugging Face.


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained("khanhld/wav2vec2-base-vietnamese-160h")
model = Wav2Vec2ForCTC.from_pretrained("khanhld/wav2vec2-base-vietnamese-160h")
model.to(device)

# Hàm chuyển đổi giọng nói thành văn bản:

def transcribe(wav_path):
    wav, _ = librosa.load(wav_path, sr=16000)
    input_values = processor(wav, sampling_rate=16000, return_tensors="pt").input_values
    logits = model(input_values.to(device)).logits
    pred_ids = torch.argmax(logits, dim=-1)
    pred_transcript = processor.batch_decode(pred_ids)[0]
    return pred_transcript

# - Gọi API ChatGPT
# - Sử dụng thư viện requests để gọi API ChatGPT.

def call_chatgpt(api_key, prompt):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]

# - Tích hợp với robot
# - Điều khiển robot đọc dữ liệu từ ChatGPT:

def robot_speak(text):
    # Đây là hàm giả lập, bạn cần thay thế bằng hàm thực sự điều khiển robot phát âm
    print(f"Robot đọc: {text}")

def main(api_key, wav_path):
    # Chuyển đổi giọng nói thành văn bản
    transcript = transcribe(wav_path)
    print(f"Nhận dạng giọng nói: {transcript}")

    # Gọi API ChatGPT với văn bản đã chuyển đổi
    response = call_chatgpt(api_key, transcript)
    print(f"ChatGPT trả lời: {response}")

    # Yêu cầu robot đọc dữ liệu
    robot_speak(response)

# API key cho ChatGPT
api_key = "your_openai_api_key"

# Đường dẫn đến file âm thanh
wav_path = "path/to/your/audio/file.wav"

main(api_key, wav_path)
