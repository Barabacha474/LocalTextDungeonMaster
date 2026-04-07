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

## USAGE

In addition to the main engine, the project includes several utility scripts located in the Codes/ directory. These scripts are intended for direct interaction with the system at the current stage of development.

### 1. main.py — Run an Adventure

This is the primary entry point of the system.

Run:

python codes/main.py

It initializes the full pipeline:

Loads the selected adventure setting
Builds the generation engine (Planner, Narrator, Memory)
Starts the console interface for interactive gameplay

You can configure:

Model names (Narrator / Planner / Summarizer)
Memory intervals
PromptCore and engine config paths
What adventure to run based on databases

See configuration section at the top of the file.

### 2. RawDataJSONLoader.py — Load a New Setting into the Database

This script is used to preprocess and load raw setting data into the vector database.

Run:

python codes/RawDataJSONLoader.py <adventure_name>

Or without arguments (interactive mode).

Requirements for input data

Your setting must follow these rules:

Located in:

SettingRawDataJSON/<adventure_name>/
Must contain:
PromptCore.json (required)
Multiple .json files with world data
Recommended practices:
Split large world information into multiple logical files
Ensure each file contains a reasonable amount of text (not too small, not too large)
Keep PromptCore.json structure unchanged (only content should be modified)
Behavior
Recursively scans all .json files
Converts each record into searchable text
Stores data in FAISS vector database
Adds metadata automatically (source file, type, etc.)
Supports batch insertion for performance

You can use vanilla_fantasy setting as guideline

### 3. DiagnosticsRunner.py — Run Tests and Verify System Health

This script runs automated tests and provides a coverage report.

Run:

python codes/DiagnosticsRunner.py
What it does

Executes all tests from:

Codes/Testers/

Measures coverage for:

Codes/Databases
Outputs a report similar to:
pytest Codes/Testers --cov=Codes/Databases --cov-report=term-missing
Purpose
Ensure database layer works correctly
Detect regressions after changes
Provide quick diagnostics before running the system
Notes
!Currently focused on database components only!
Can be extended to cover other modules in future

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

