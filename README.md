# AI Adventure Engine (Local LLM-based Narrative System)

An interactive adventure engine powered by local Large Language Models (LLMs).
The system simulates a structured storytelling pipeline with multiple AI roles such as Planner, Narrator, and Summarizer, enabling dynamic, controllable, and extensible story generation.

This project is designed as a diploma-level system demonstrating multi-agent orchestration, deterministic pipelines, and local AI execution without reliance on cloud services.

---

## Features

- Adventure planning system with structured progression logic (Can be disabled for better generation speed)
- Command-based interaction for triggering generation steps
- Commands functionality for rolling back narrative steps
- Local LLM execution via Ollama
- Support for multiple models assigned to different roles
- Flexible setting system:
  - Any world or setting can be defined via JSON configuration files
  - Supports highly detailed environments through structured descriptions
  - Designed for compatibility with custom instruction sets (expandable)

---

## Requirements

- Python 3.10+
- Ollama installed
- A downloaded LLM model (e.g. cogito:8b-v1-preview-llama-q4_K_M), that will fit in your device capabilities. (model and context stored in GPU vram and CPU ram during generation. CPU generation is usually way slower.)

---

## Installation

### 1. Install Ollama

Download and install Ollama from the official website:
https://ollama.com/download

- On Windows/macOS: run the installer
- On Linux:
curl -fsSL https://ollama.com/install.sh | sh

After installation, make sure Ollama is running.

---

### 2. Download a model

Pull a model using the CLI:

ollama pull <your-model-name>

Example:
cogito:8b-v1-preview-llama-q4_K_M

You can use any compatible model available in Ollama.

---

### 3. Clone the repository

git clone https://github.com/Barabacha474/LocalTextDungeonMaster.git
cd your-repo-name

---

### 4. Configure model names

Open:
codes/main.py

Set the model names you want to use:

NARRATOR_MODEL_NAME: str = "_your-model-name_"

PLANNER_MODEL_NAME: str = "_your-model-name_"

SUMMARIZER_MODEL_NAME: str = "_your-model-name_"


---

## Running the Project

Run the main script:

python codes/main.py

The system will start generating an interactive adventure using your local LLM.

---

## How It Works

The engine operates as a structured generation pipeline:

1. Planner
   Determines the next step in the adventure based on current context and previous plans

2. Narrator
   Generates the narrative content

3. Summarizer
   Maintains a compressed representation of the story state

Other systems designed to collect, store and provide context for LLMs and logs for researchers.
---

## Notes

- Performance depends on your hardware (CPU/GPU and RAM)
- Larger models produce better results but require more resources
- All inference runs locally through Ollama
- No internet connection is required after setup
- Project is in active development and bugs may occure

---

## License

MIT License

Copyright (c) 2026 Barabacha474

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

