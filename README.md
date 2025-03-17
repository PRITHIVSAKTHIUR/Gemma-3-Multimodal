![gemma-3_2 original](https://github.com/user-attachments/assets/a79797dd-116b-43e0-bc4b-5977e03d8f59)

# Gemma 3 Multimodal Chat Interface

This project provides a multimodal chat interface powered by the **Gemma 3** model, capable of handling text, image, and video inputs. The interface allows users to interact with the model using natural language queries, and it supports advanced features like video understanding and OCR (Optical Character Recognition). The system is built using **Gradio** for the user interface and **Hugging Face Transformers** for model inference.

## Features

- **Text Generation**: Generate text responses based on user input using the **FastThink-0.5B-Tiny** model.
- **Multimodal Input**: Supports image and video inputs for tasks like image captioning, video understanding, and OCR.
- **Gemma 3 Integration**: Utilizes the **Gemma 3** model for advanced text and multimodal tasks.
  - **`@gemma3`**: Use this flag to invoke the Gemma 3 model for text and image-based queries.
  - **`@video-infer`**: Use this flag to analyze and understand video content.
- **Video Processing**: Downsample videos to extract key frames for analysis.
- **Customizable Parameters**: Adjust generation parameters like `max_new_tokens`, `temperature`, `top-p`, `top-k`, and `repetition_penalty`.
- **Real-time Streaming**: Stream responses in real-time for a seamless user experience.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/prithivsakthiur/gemma3-multimodal-chat.git
   cd gemma3-multimodal-chat
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the interface**:
   - The application will launch a Gradio interface. You can access it via the provided local or public URL.

## Usage

### Text Generation
- Enter your text query in the input box.
- The model will generate a response based on the input.

### Image-Based Queries
- Use the `@gemma3` flag followed by your query.
- Upload one or more images.
- The model will analyze the images and generate a response.

### Video-Based Queries
- Use the `@video-infer` flag followed by your query.
- Upload a video file.
- The model will extract key frames from the video, analyze them, and generate a response.

### Customizing Generation Parameters
- Adjust the following parameters to control the generation process:
  - **Max new tokens**: Controls the length of the generated response.
  - **Temperature**: Controls the randomness of the output.
  - **Top-p (nucleus sampling)**: Controls the diversity of the output.
  - **Top-k**: Limits the sampling pool to the top-k tokens.
  - **Repetition penalty**: Penalizes repeated tokens to reduce redundancy.

## Examples

1. **Text Generation**:
   - Input: `Python Program for Array Rotation`
   - Output: The model generates a Python program for rotating an array.

2. **Image-Based Query**:
   - Input: `@gemma3 Explain the Image`
   - Upload: An image file (e.g., `examples/3.jpg`)
   - Output: The model provides a detailed explanation of the image content.

3. **Video-Based Query**:
   - Input: `@video-infer Explain the content of the Advertisement`
   - Upload: A video file (e.g., `examples/videoplayback.mp4`)
   - Output: The model analyzes the video and explains its content.

## Models Used

- **FastThink-0.5B-Tiny**: A lightweight text generation model for general-purpose text tasks.
- **Qwen2-VL-OCR-2B-Instruct**: A multimodal model for OCR and image understanding.
- **Gemma 3**: A powerful multimodal model for text, image, and video understanding.

## Environment Variables

- `MAX_INPUT_TOKEN_LENGTH`: Maximum allowed input token length (default: 4096).
- `MAX_IMAGE_SIZE`: Maximum image size (default: 2048).
- `CACHE_EXAMPLES`: Enable example caching (default: 0).
- `USE_TORCH_COMPILE`: Enable Torch compilation (default: 0).
- `ENABLE_CPU_OFFLOAD`: Enable CPU offloading (default: 0).
- `BAD_WORDS`: List of bad words to filter (default: []).
- `BAD_WORDS_NEGATIVE`: List of negative bad words to filter (default: []).
- `default_negative`: Default negative prompt (default: "").

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
