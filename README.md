# Coraline Theory Video Creator

![Project Banner](placeholder_banner.png)

> **An autonomous multimodal AI system that watches the movie *Coraline*, analyzes it for specific user-defined theories (e.g., "The Hidden Entomology"), and produces a fully edited video essay—completely on its own.**

This project demonstrates an end-to-end AI pipeline that combines computer vision, natural language processing, and automated video editing to generate high-quality documentary content without human intervention.

## 🎥 Example Output: "The Hidden Entomology of Coraline"

[![Watch the Example Video](https://img.youtube.com/vi/KVWmnIjI2b0/0.jpg)](https://www.youtube.com/watch?v=KVWmnIjI2b0)

This tool was used to generate a full analysis on the entomological motifs in the film. The system autonomously:
*   Identified insect-related scenes using CLIP.
*   Generated a script connecting bugs to the Beldam's trap.
*   Created animations and voiceovers to explain the theory.

## 🚀 Key Features

*   **Autonomous Narrative Synthesis**: Uses **Gemini 2.1 Flash** to analyze film semiotics and generate a structured script correlated with visual themes.
*   **Theory-Driven Analysis**: Can be prompted with any theory (e.g., "Color Psychology", "The Beldam's Origin") to generate a specific video essay.
*   **Semantic Scene Retrieval**: Implements a **CLIP-based Vector Search** engine to find exact film segments matching the script's thematic content.
*   **High-Performance Animation Engine**: A custom **Multi-Threaded Rendering System** (built with PIL & NumPy) that composites animated character overlays at 60fps, 40x faster than real-time.
*   **Automated Editing Pipeline**: Orchestrates **FFmpeg**, **Whisper ASR**, and **Kokoro TTS** to synchronize narration, subtitles, and background music with frame-perfect precision.
*   **Hybrid AI Architecture**:leverages both Cloud LLMs (Gemini) for complex reasoning and Local Models (Ollama/Qwen) for rapid decision-making.

## 🛠️ Tech Stack

*   **Core Framework**: Python, Streamlit
*   **AI Models**:
    *   **LLM**: Google Gemini 2.1 Flash, Qwen 2.5 (via Ollama)
    *   **Vision**: CLIP (Contrastive Language-Image Pre-Training), OpenCV
    *   **Audio**: OpenAI Whisper (ASR), Kokoro TTS (Voice Synthesis)
*   **Video Processing**: FFmpeg, MoviePy, PIL (Pillow)
*   **Tools**: Selenium (for web asset retrieval), NumPy, Pandas

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/23f2001390/coraline-theory-video-creator.git
    cd coraline-theory-video-creator
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup API Keys**
    Create a `api_keys_config.json` file in the root directory:
    ```json
    {
        "gemini_api_key": "YOUR_GEMINI_API_KEY",
        "pexels_api_key": "YOUR_PEXELS_API_KEY" (Optional)
    }
    ```

4.  **Download Models**
    *   Ensure `kokoro-v1.0.onnx` and `voices-v1.0.bin` are placed in the root directory.
    *   (These are large files and might need to be downloaded separately if not included in the repo).

## 🎮 Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

1.  **Select Source**: Choose the *Coraline* movie file or provide a YouTube URL.
2.  **Configure Analysis**: Set the analysis depth and focus (e.g., "Entomology", "Color Theory").
3.  **Generate**: Click "Start Autonomous Creation". The system will:
    *   Deconstruct the film into scenes.
    *   Generate a script.
    *   Match scenes to the script.
    *   Synthesize voiceover.
    *   Render animations.
    *   Compile the final video.

## 📂 Project Structure

```
├── app.py                 # Main application verified entry point
├── character_assets/      # Character sprites for animations
├── avatar_assets/         # Avatar images for UI
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
