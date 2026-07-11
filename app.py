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

# ----------------- SCIKIT-LEARN MODEL SETUP -----------------
# We train a simple linear regression model simulating historical rental quotes.
# Features: [crowd_count, venue_size_sqm]
X_train = np.array([
    [100, 50], [200, 100], [300, 150], [500, 250], [1000, 500],
    [150, 75], [250, 120], [400, 200], [800, 400], [1200, 600]
])
# Labels: Base equipment cost in USD
y_train = np.array([
    1200, 2200, 3100, 4900, 9500,
    1600, 2600, 4100, 7800, 11500
])

pricing_model = LinearRegression()
pricing_model.fit(X_train, y_train)

def predict_fair_price(crowd_count: int, venue_size_sqm: int) -> float:
    """
    Predict the fair market rental price for equipment based on event parameters.
    Uses a Scikit-Learn Linear Regression model trained on historical data.
    """
    try:
        prediction = pricing_model.predict([[crowd_count, venue_size_sqm]])
        # Ensure price is a positive number
        return float(max(500.0, round(prediction[0], 2)))
    except Exception as e:
        print(f"Error in Scikit-Learn prediction: {e}")
        return 1500.0  # Fallback

# ----------------- STANDARD AV SETUP CHECKLIST -----------------
def get_standard_equipment(event_type: str) -> str:
    """
    Get standard equipment recommendations based on the event type.
    """
    event_type = event_type.lower()
    if "concert" in event_type or "show" in event_type or "musical" in event_type:
        return "Concert setups require: Line array speakers, high-power subwoofers, stage lighting (LED pars, moving heads), stage monitors, professional audio mixer (digital), and vocal/instrument microphones."
    elif "wedding" in event_type or "reception" in event_type:
        return "Wedding setups require: Column speakers or standard PA speakers, ambient/uplighting, wireless lapel & handheld microphones, background music player, and subtle stage lighting."
    elif "conference" in event_type or "corporate" in event_type or "seminar" in event_type:
        return "Corporate/Conference setups require: PA speakers, wireless lapel microphones, visual display (LED screens or projectors), podium microphone, and presenter clicker."
    else:
        return "Standard event setups require: Basic PA sound system, wireless microphones, general ambient lighting, and essential power cabling."

# ----------------- AI AGENT SYSTEM -----------------
@app.route('/api/generate', methods=['POST'])
def generate_infrastructure_plan():
    """
    POST /api/generate
    Expects JSON: { event_type, crowd_count, venue_size_sqm, budget_range }
    Acts as an autonomous AI Agent that makes decisions, runs local models (Scikit-Learn),
    and outputs a structured equipment plan.
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

    try:
        # Step 1: Run the local Scikit-Learn model to predict the fair price
        predicted_cost = predict_fair_price(crowd_count, venue_size_sqm)

        # Step 2: Fetch the standard equipment checklist for the event type
        std_checklist = get_standard_equipment(event_type)

        # Step 3: Run the Agentic workflow with Gemini
        # We instruct Gemini to act as an expert AV consultant, review the ML pricing estimate,
        # adjust equipment recommendations to fit the budget range, and output raw JSON.
        prompt = f"""
        You are the SoundScout AI Agent, an expert Event Infrastructure Consultant.
        
        Event Details:
        - Type: {event_type}
        - Expected Crowd: {crowd_count} people
        - Venue Size: {venue_size_sqm} sqm
        - User Budget Range: {budget_range}
        
        Information Gathered by Agent:
        1. Standard AV Checklist: "{std_checklist}"
        2. Scikit-Learn ML Predicted Fair Cost: ${predicted_cost:.2f}
        
        Task:
        1. Formulate a final, tailored equipment plan representing the items needed.
        2. Adjust quantities and quality of the equipment to fit the User Budget Range if needed, while ensuring safety and minimum coverage.
        3. Output the result strictly as a JSON array of strings under the key "equipment_plan". Do not output markdown, formatting, or extra text.

        Example Output format:
        {{
            "equipment_plan": [
                "2x Line Array Speakers",
                "1x 16-Channel Mixer",
                "4x Wireless Handheld Microphones"
            ]
        }}
        """

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json"}
        )

        response = model.generate_content(prompt)
        
        # Parse the JSON response
        result_json = json.loads(response.text)
        
        # Add metadata showing the agent's calculations for the pitch
        result_json["ml_predicted_cost"] = predicted_cost
        result_json["scoped_budget_range"] = budget_range

        return jsonify(result_json), 200

    except Exception as e:
        print(f"AI Agent execution failed: {e}")
        return jsonify({"error": "Internal AI Agent error"}), 500

if __name__ == '__main__':
    # Run the service on port 8000
    app.run(host='0.0.0.0', port=8000, debug=True)
