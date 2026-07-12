import os
import json
from flask import Flask, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
import numpy as np

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("⚠️ Warning: GEMINI_API_KEY is not set in environment variables.")

# ----------------- SCIKIT-LEARN PRICING MODEL -----------------
# Simple regression model trained on historical AV logistics quotes
X_train = np.array([
    [100, 50], [200, 100], [300, 150], [500, 250], [1000, 500],
    [150, 75], [250, 120], [400, 200], [800, 400], [1200, 600]
])
y_train = np.array([
    1200, 2200, 3100, 4900, 9500,
    1600, 2600, 4100, 7800, 11500
])

pricing_model = LinearRegression()
pricing_model.fit(X_train, y_train)

def predict_fair_price(crowd_count: int, venue_size_sqm: int) -> float:
    """Predicts a baseline fair market price using Scikit-Learn."""
    try:
        prediction = pricing_model.predict([[crowd_count, venue_size_sqm]])
        return float(max(500.0, round(prediction[0], 2)))
    except Exception as e:
        print(f"Pricing Model Error: {e}")
        return 1500.0

# ----------------- LOGISTICS POWER CALCULATOR (Deterministic Tool) -----------------
def calculate_power_needs(audio_items_count: int, visual_items_count: int) -> dict:
    """
    Deterministic mathematical utility to compute power load safely (no LLM math errors).
    """
    # Baseline power estimates per item type in Kilowatts
    audio_draw_kw = audio_items_count * 0.8  # e.g., 800W per speaker/amp
    visual_draw_kw = visual_items_count * 1.5 # e.g., high-power LED panels / moving heads
    
    total_draw = audio_draw_kw + visual_draw_kw
    suggested_generator = "25kVA Generator"
    if total_draw > 20:
        suggested_generator = "100kVA Generator"
    elif total_draw > 10:
        suggested_generator = "50kVA Generator"
        
    return {
        "total_draw_kw": round(total_draw, 2),
        "suggested_generator": suggested_generator
    }

# ----------------- MULTI-AGENT PIPELINE -----------------
@app.route('/api/generate', methods=['POST'])
def generate_infrastructure_plan():
    """
    POST /api/generate
    Coordinates a sequential multi-agent AV design pipeline:
    1. Audio Specialist Agent (Flash)
    2. Visual/Lighting Specialist Agent (Flash)
    3. Power & Logistics Specialist Agent (Flash)
    4. Lead Coordinator Agent (Pro) + ML Pricing Tool + Power Calculator Tool
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    event_type = data.get("event_type", "General Event")
    crowd_count = int(data.get("crowd_count", 100))
    venue_size_sqm = int(data.get("venue_size_sqm", 50))
    budget_range = data.get("budget_range", "Unknown")

    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API key is not configured on AI service."}), 500

    # Instantiate Gemini models (Flash for workers, Pro for coordinator/reasoning)
    try:
        model_flash = genai.GenerativeModel("gemini-1.5-flash")
        model_pro = genai.GenerativeModel("gemini-1.5-pro")
    except Exception as e:
        return jsonify({"error": f"Failed to load models: {str(e)}"}), 500

    try:
        # === AGENT 1: Audio Specialist Agent (Flash) ===
        print(f"[Agent 1: Audio Specialist] Analyzing audio requirements for {crowd_count} people...")
        audio_prompt = f"""
        You are the SoundScout Audio Specialist Agent.
        Analyze the audio requirements for this event:
        - Event Type: {event_type}
        - Crowd Count: {crowd_count} people
        - Venue Size: {venue_size_sqm} sqm

        Recommend specific audio hardware (speakers, subwoofers, mixers, microphones) including exact quantities.
        Provide a concise, professional list and explanation of the audio choices.
        """
        audio_response = model_flash.generate_content(audio_prompt).text
        print("✅ [Agent 1] Completed audio recommendation.")

        # === AGENT 2: Visual & Lighting Specialist Agent (Flash) ===
        print(f"[Agent 2: Visual Specialist] Analyzing lighting & screens for {venue_size_sqm} sqm...")
        visual_prompt = f"""
        You are the SoundScout Visual and Lighting Specialist Agent.
        Analyze the lighting and display requirements for this event:
        - Event Type: {event_type}
        - Crowd Count: {crowd_count} people
        - Venue Size: {venue_size_sqm} sqm

        Recommend visual displays (LED walls, projectors) and lighting equipment (LED Pars, moving heads, lasers) with exact quantities.
        Provide a concise, professional list.
        """
        visual_response = model_flash.generate_content(visual_prompt).text
        print("✅ [Agent 2] Completed visual recommendation.")

        # === DETERMINISTIC POWER CALCULATOR TOOL ===
        # Count items from text to feed the calculator (approximate heuristic)
        audio_items_est = audio_response.count("\n") + 5
        visual_items_est = visual_response.count("\n") + 5
        power_calc = calculate_power_needs(audio_items_est, visual_items_est)

        # === AGENT 3: Power & Logistics Agent (Flash) ===
        print(f"[Agent 3: Logistics Specialist] Calculating power load and staging...")
        logistics_prompt = f"""
        You are the SoundScout Power and Staging Logistics Agent.
        Review the following equipment list recommendations:
        
        Audio Plan:
        {audio_response}
        
        Visual Plan:
        {visual_response}

        Deterministic Calculations Provided:
        - Estimated power draw: {power_calc['total_draw_kw']} kW
        - Suggested Generator: {power_calc['suggested_generator']}

        Formulate a staging, cabling, and power distribution plan. Reconfirm if the suggested generator is correct or if a backup generator is needed.
        """
        logistics_response = model_flash.generate_content(logistics_prompt).text
        print("✅ [Agent 3] Completed logistics recommendation.")

        # === TOOL: Query Scikit-Learn Model ===
        predicted_cost = predict_fair_price(crowd_count, venue_size_sqm)

        # === AGENT 4: Lead Coordinator Agent (Pro) ===
        print(f"[Agent 4: Lead Coordinator] Consolidating plan to match budget {budget_range}...")
        coordinator_prompt = f"""
        You are the Lead Coordinator Agent (AV Technical Director) for SoundScout AI.
        
        You must review the specialized plans from your sub-agents:
        - Audio Specialist Plan: {audio_response}
        - Visual Specialist Plan: {visual_response}
        - Logistics Plan: {logistics_response}
        
        Constraints:
        - User's Budget: {budget_range}
        - ML Predicted Base Market Cost: ${predicted_cost:.2f}
        
        Your Task:
        1. Consolidate these plans into a single coherent equipment list.
        2. Adjust the quantities and models of equipment so that the plan realistically fits the user's budget range. If the budget is low, prioritize essential audio and basic lighting. If high, add premium subwoofers and larger screens.
        3. Output the result STRICTLY as a JSON array of strings under the key "equipment_plan". Do not output markdown, notes, or extra text.

        Example Output format:
        {{
            "equipment_plan": [
                "2x Line Array Speakers (High Quality)",
                "1x 16-Channel Mixer",
                "4x Stage Monitor Wedges",
                "1x 50kVA Generator (Power)"
            ]
        }}
        """
        
        # We use Pro here for high-level synthesis, budget matching, and JSON structure adherence
        coordinator_response = model_pro.generate_content(
            coordinator_prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        result_json = json.loads(coordinator_response.text)
        
        # Add metadata for demonstration / auditing
        result_json["ml_predicted_cost"] = predicted_cost
        result_json["scoped_budget_range"] = budget_range
        result_json["calculated_power_draw_kw"] = power_calc["total_draw_kw"]
        result_json["agent_spec_log"] = {
            "audio_summary": audio_response[:300] + "...",
            "visual_summary": visual_response[:300] + "...",
            "logistics_summary": logistics_response[:300] + "..."
        }

        print("✅ [Agent 4] Final consolidated plan generated successfully.")
        return jsonify(result_json), 200

    except Exception as e:
        print(f"Multi-Agent Execution Failed: {e}")
        return jsonify({"error": f"Internal AI Agent error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
