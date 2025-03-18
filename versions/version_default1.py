import os
import random
import uuid
import json
import time
import asyncio
import re
from threading import Thread

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import cv2

from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    TextIteratorStreamer,
)
from transformers.image_utils import load_image

# Constants
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))
MAX_SEED = np.iinfo(np.int32).max

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Helper function to return a progress bar HTML snippet.
def progress_bar_html(label: str) -> str:
    return f'''
<div style="display: flex; align-items: center;">
    <span style="margin-right: 10px; font-size: 14px;">{label}</span>
    <div style="width: 110px; height: 5px; background-color: #F0FFF0; border-radius: 2px; overflow: hidden;">
        <div style="width: 100%; height: 100%; background-color: #00FF00; animation: loading 1.5s linear infinite;"></div>
    </div>
</div>
<style>
@keyframes loading {{
    0% {{ transform: translateX(-100%); }}
    100% {{ transform: translateX(100%); }}
}}
</style>
    '''

# Qwen2-VL (for optional image inference)

MODEL_ID_VL = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct" 
processor = AutoProcessor.from_pretrained(MODEL_ID_VL, trust_remote_code=True)
model_m = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID_VL,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to("cuda").eval()

def clean_chat_history(chat_history):
    cleaned = []
    for msg in chat_history:
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            cleaned.append(msg)
    return cleaned

bad_words = json.loads(os.getenv('BAD_WORDS', "[]"))
bad_words_negative = json.loads(os.getenv('BAD_WORDS_NEGATIVE', "[]"))
default_negative = os.getenv("default_negative", "")

def check_text(prompt, negative=""):
    for i in bad_words:
        if i in prompt:
            return True
    for i in bad_words_negative:
        if i in negative:
            return True
    return False

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "0") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2048"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

dtype = torch.float16 if device.type == "cuda" else torch.float32


# Gemma3 Model (default for text, image, & video inference)

gemma3_model_id = "google/gemma-3-4b-it"  # alternative: google/gemma-3-12b-it
gemma3_model = Gemma3ForConditionalGeneration.from_pretrained(
    gemma3_model_id, device_map="auto"
).eval()
gemma3_processor = AutoProcessor.from_pretrained(gemma3_model_id)

# VIDEO PROCESSING HELPER

def downsample_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    # Sample 10 evenly spaced frames.
    frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
    for i in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            # Convert from BGR to RGB and then to PIL Image.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            timestamp = round(i / fps, 2)
            frames.append((pil_image, timestamp))
    vidcap.release()
    return frames

# MAIN GENERATION FUNCTION

@spaces.GPU
def generate(
    input_dict: dict,
    chat_history: list[dict],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
):
    text = input_dict["text"]
    files = input_dict.get("files", [])
    lower_text = text.lower().strip()

    # ----- Qwen2-VL branch (triggered with @qwen2-vl) -----
    if lower_text.startswith("@qwen2-vl"):
        prompt_clean = re.sub(r"@qwen2-vl", "", text, flags=re.IGNORECASE).strip().strip('"')
        if files:
            images = [load_image(f) for f in files]
            messages = [{
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in images],
                    {"type": "text", "text": prompt_clean},
                ]
            }]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=images, return_tensors="pt", padding=True).to("cuda")
        else:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt_clean}]}
            ]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to("cuda", dtype=torch.float16)
        streamer = TextIteratorStreamer(processor.tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }
        thread = Thread(target=model_m.generate, kwargs=generation_kwargs)
        thread.start()
        buffer = ""
        yield progress_bar_html("Processing with Qwen2VL")
        for new_text in streamer:
            buffer += new_text
            buffer = buffer.replace("<|im_end|>", "")
            time.sleep(0.01)
            yield buffer
        return

    # ----- Default branch: Gemma3 (for text, image, & video inference) -----
    if files:
        # Check if any provided file is a video based on extension.
        video_extensions = (".mp4", ".mov", ".avi", ".mkv", ".webm")
        if any(str(f).lower().endswith(video_extensions) for f in files):
            # Video inference branch.
            prompt_clean = re.sub(r"@video-infer", "", text, flags=re.IGNORECASE).strip().strip('"')
            video_path = files[0]
            frames = downsample_video(video_path)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt_clean}]}
            ]
            # Append each frame (with its timestamp) to the conversation.
            for frame in frames:
                image, timestamp = frame
                image_path = f"video_frame_{uuid.uuid4().hex}.png"
                image.save(image_path)
                messages[1]["content"].append({"type": "text", "text": f"Frame {timestamp}:"})
                messages[1]["content"].append({"type": "image", "url": image_path})
            inputs = gemma3_processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(gemma3_model.device, dtype=torch.bfloat16)
            streamer = TextIteratorStreamer(gemma3_processor.tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            thread = Thread(target=gemma3_model.generate, kwargs=generation_kwargs)
            thread.start()
            buffer = ""
            yield progress_bar_html("Processing video with Gemma3")
            for new_text in streamer:
                buffer += new_text
                time.sleep(0.01)
                yield buffer
            return
        else:
            # Image inference branch.
            prompt_clean = re.sub(r"@gemma3", "", text, flags=re.IGNORECASE).strip().strip('"')
            images = [load_image(f) for f in files]
            messages = [{
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in images],
                    {"type": "text", "text": prompt_clean},
                ]
            }]
            inputs = gemma3_processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            ).to(gemma3_model.device, dtype=torch.bfloat16)
            streamer = TextIteratorStreamer(gemma3_processor.tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            thread = Thread(target=gemma3_model.generate, kwargs=generation_kwargs)
            thread.start()
            buffer = ""
            yield progress_bar_html("Processing with Gemma3")
            for new_text in streamer:
                buffer += new_text
                time.sleep(0.01)
                yield buffer
            return
    else:
        # Text-only inference branch.
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": text}]}
        ]
        inputs = gemma3_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(gemma3_model.device, dtype=torch.bfloat16)
        streamer = TextIteratorStreamer(gemma3_processor.tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }
        thread = Thread(target=gemma3_model.generate, kwargs=generation_kwargs)
        thread.start()
        outputs = []
        for new_text in streamer:
            outputs.append(new_text)
            yield "".join(outputs)
        final_response = "".join(outputs)
        yield final_response


# Gradio Interface

demo = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS),
        gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6),
        gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9),
        gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50),
        gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2),
    ],
    examples=[
        [
            {
                "text": "Create a short story based on the images.",
                "files": [
                    "examples/1111.jpg",
                    "examples/2222.jpg",
                    "examples/3333.jpg",
                ],
            }
        ],
        [{"text": "Explain the Image", "files": ["examples/3.jpg"]}],
        [{"text": "Explain the content of the Advertisement", "files": ["examples/videoplayback.mp4"]}],
        [{"text": "Which movie character is this?", "files": ["examples/9999.jpg"]}],
        ["Explain Critical Temperature of Substance"],
        [{"text": "@qwen2-vl Transcription of the letter", "files": ["examples/222.png"]}],
        [{"text": "Explain the content of the video in detail", "files": ["examples/breakfast.mp4"]}],
        [{"text": "Describe the video", "files": ["examples/Missing.mp4"]}],
        [{"text": "Explain what is happening in this video ?", "files": ["examples/oreo.mp4"]}],
        [{"text": "Summarize the events in this video", "files": ["examples/sky.mp4"]}],
        [{"text": "What is in the video ?", "files": ["examples/redlight.mp4"]}],
        ["Python Program for Array Rotation"],
        ["Explain Critical Temperature of Substance"]
    ],
    cache_examples=False,
    type="messages",
    description="# **Gemma 3 Multimodal** \n`Use @qwen2-vl to switch to Qwen2-VL OCR for image inference and @video-infer for video input`",
    fill_height=True,
    textbox=gr.MultimodalTextbox(label="Query Input", file_types=["image", "video"], file_count="multiple", placeholder="Tag with @qwen2-vl for Qwen2-VL inference if needed."),
    stop_btn="Stop Generation",
    multimodal=True,
)

if __name__ == "__main__":
    demo.queue(max_size=20).launch(share=True)
