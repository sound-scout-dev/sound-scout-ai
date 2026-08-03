# 🎵 SoundScout AI Microservice

The dedicated AI proxy and multi-agent intelligence microservice. It handles complex natural language processing, event logistics decomposition, vendor requirement scoring, and conversational AI support.

## 🛠️ Tech Stack
* **Runtime:** Python 3.11 / FastAPI 
* **AI Models:** Gemini 
* **HTTP Engine:** Uvicorn / requests
* **Data Validation:** Pydantic / JSON Schema validation

## 🚀 Core AI Responsibilities
* **Event Decomposition Engine (`/api/ai/decompose`):** Accepts raw event prompts (e.g., "Indoor acoustic concert for 300 guests") and outputs structured JSON containing precise wattage, channel counts, and lighting requirements.
* **Vendor Matching (`/api/ai/recommend`):** Analyzes vendor proposals against event logistics criteria and scores the top recommended bids.
* **Conversational Support (`/api/ai/chat`):** Generates real-time, context-aware responses for organizers chatting via WhatsApp.

## ⚙️ Setup & Execution

### Prerequisites
* Python 3.11+

### Installation
```bash
git clone [https://github.com/sound-scout-dev/sound-scout-ai.git](https://github.com/sound-scout-dev/sound-scout-ai.git)
cd sound-scout-ai
pip install -r requirements.txt
```

### Environment Variables (`.env`)
Create a `.env` file in the root directory and add:
```env
PORT=8000
OPENROUTER_API_KEY=your_openrouter_key
GEMINI_API_KEY=your_gemini_key
```

### Run Locally
```bash
python main.py
# or using uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
