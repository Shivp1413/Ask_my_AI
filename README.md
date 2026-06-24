# Ask My AI 

**Ask_my_AI** is an open-source, secure, privacy-first local interface for running vision-language models such as **Qwen3-VL** and **SmolVLM** entirely on your own hardware.

<p align="center">
  <img src="https://github.com/user-attachments/assets/49f429ac-7221-4fc1-90dd-656a9ecb1433" width="220" />
  <img src="https://github.com/user-attachments/assets/0ad14149-4e65-4d80-944b-b2d76a2492eb" width="220" />
  <img src="https://github.com/user-attachments/assets/f87301c2-86c9-4841-9ff2-311f811cb803" width="220" />
  <img src="https://github.com/user-attachments/assets/30693f30-d5cd-4d9d-af05-b451943cc302" width="220" />
</p>


It can be used on a wide range of devices; however, the current repository is optimized for the **Raspberry Pi 5 (4 GB variant)**.

## Installation

Clone the repository:

```bash
https://github.com/Shivp1413/Ask_my_AI.git
cd Ask_my_AI
```

Follow the setup instructions provided in the repository, then start the application by running:

```bash
python3 main.py
```

Repository:

https://github.com/Shivp1413/Ask_my_AI


#  RPi 5 Real-Time Vision with SmolVLM/Qwen3.5:0.8B + Streamlit Command Sheet


---

## **Step 1 — System update**
```bash
sudo apt update && sudo apt upgrade -y && sudo apt install -y build-essential cmake git python3-pip python3-venv libcamera-dev
```

---

## **Step 2 — Create 6 GB swap (one-liner)**
```bash
sudo swapoff -a && sudo fallocate -l 6G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile && echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Check: `free -h`

---

## **Step 3 — Build llama.cpp (with server + multimodal)**
```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=ON
cmake --build build -j4 --config Release
sudo cp build/bin/llama-* /usr/local/bin/
```

---

## **Step 4 — Download both models from HuggingFace**
```bash
mkdir -p ~/models/smolvlm && cd ~/models/smolvlm && \
curl -L -O https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf && \
curl -L -O https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf


#verify the download
ls -lh ~/models/smolvlm/

mkdir -p ~/models/qwen && cd ~/models/qwen && \
curl -L -O https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf

cd ~/models/smolvlm && \
curl -L -O https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf && \
curl -L -O https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf
```

**Start the vision server (in one terminal, keep it running):**
```bash
llama-server \
  -m ~/models/smolvlm/SmolVLM-500M-Instruct-Q8_0.gguf \
  --mmproj ~/models/smolvlm/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
  --host 0.0.0.0 --port 8080 -c 2048 -t 4
```

(Qwen text-only server, optional, on port 8081:)
```bash
llama-server -m ~/models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf --host 0.0.0.0 --port 8081 -c 2048 -t 4
```

---

## **Step 5 — Streamlit UI (real-time camera → SmolVLM)**

Install once:
```bash
pip install streamlit streamlit-webrtc opencv-python-headless av requests
```


Run it (must be HTTPS for iPhone camera — use this trick):
```bash
streamlit run main.py --server.address 0.0.0.0 --server.port 8501
```

On iPhone Chrome go to:  
👉 `http://<rpi-ip>:8501`

> ⚠️ iPhone Safari/Chrome requires **HTTPS** for camera access. Quickest fix — install **`stunnel`** or run through **ngrok**:
> ```bash
> # one-time
> curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
> echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
> sudo apt update && sudo apt install ngrok
> ngrok http 8501
> ```
> Then open the **https://xxxx.ngrok.app** URL on the iPhone → camera works ✅

---

### What you get
- Backend prints each caption in terminal (`>> a man holding a coffee cup`)
- Streamlit UI shows live video with overlay + big text caption updated every ~1 s
- Uses only ~1.2 GB RAM (SmolVLM-500M Q8) → fits your 4 GB RPi 5 easily
