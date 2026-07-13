# ai/app.py
import os
import json
import time
from flask import Flask, request, jsonify
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Gemini Client using the modern google-genai library
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Error initializing GenAI Client: {e}")
else:
    print("⚠️ Warning: GEMINI_API_KEY is not set in environment variables.")

# ----------------- SCIKIT-LEARN PRICING MODEL -----------------
def generate_sri_lankan_synthetic_data(num_samples=250):
    """
    Generates a realistic synthetic dataset representing historical Sri Lankan AV rental quotes.
    Features: [crowd_count, venue_size_sqm]
    """
    np.random.seed(42)
    # Generate crowd sizes from tiny university seminars (20 people) to massive concerts (2000 people)
    crowds = np.random.randint(20, 2000, size=num_samples)
    # Generate venue sizes: roughly proportional to crowd size (0.5 to 1.5 sqm per person)
    venue_sizes = crowds * np.random.uniform(0.5, 1.5, size=num_samples)
    venue_sizes = np.clip(venue_sizes, 10, 1500).astype(int)
    
    prices = []
    for crowd, size in zip(crowds, venue_sizes):
        # Base setup + logistics handling (min price)
        base = 8000
        base += crowd * 70  # cost scaling per person
        base += size * 120  # cost scaling per venue size
        
        # Step-wise gear additions (non-linear scaling)
        if crowd > 1000:
            base += 450000   # Large concert line arrays, rigging, heavy power
        elif crowd > 500:
            base += 180000   # Double subwoofers, trussing, moving heads
        elif crowd > 150:
            base += 50000    # Mid-range audio, monitors, basic lighting
        elif crowd > 50:
            base += 15000    # Extra mics, ambient lighting
            
        # Add random noise to simulate market variance / client negotiation (+/- 12%)
        noise = np.random.uniform(-0.12, 0.12) * base
        
        # Ensure minimum LKR budget floor starts at LKR 10,000 (university friendly)
        price = max(10000.0, round(base + noise, -3))
        prices.append(price)
        
    X = np.column_stack((crowds, venue_sizes))
    y = np.array(prices)
    return X, y

# Generate data and train a Random Forest Regressor (great for stepwise non-linear pricing)
print("⚙️ Generating historical Sri Lankan AV rental dataset...")
X_train, y_train = generate_sri_lankan_synthetic_data(250)
print(f"📊 Training Random Forest model on {len(X_train)} quotes...")
pricing_model = RandomForestRegressor(n_estimators=100, random_state=42)
pricing_model.fit(X_train, y_train)
print("✅ Random Forest pricing model trained and online.")

def predict_fair_price(crowd_count: int, venue_size_sqm: int) -> float:
    """Predicts a baseline fair market price in Sri Lankan Rupees (LKR)."""
    try:
        prediction = pricing_model.predict([[crowd_count, venue_size_sqm]])
        # Set absolute lowest baseline LKR budget floor to LKR 10,000
        return float(max(10000.0, round(prediction[0], 2)))
    except Exception as e:
        print(f"Pricing Model Error: {e}")
        return 120000.0  # Fallback LKR

# ----------------- LOGISTICS POWER CALCULATOR (Deterministic Tool) -----------------
def calculate_power_needs(audio_items_count: int, visual_items_count: int) -> dict:
    """
    Deterministic mathematical utility to compute power load safely (no LLM math errors).
    """
    audio_draw_kw = audio_items_count * 0.8  
    visual_draw_kw = visual_items_count * 1.5 
    
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

# ----------------- RETRY HELPER FOR TRANSIENT API ERRORS -----------------
def generate_content_with_retry(model_name, contents, config=None, max_retries=4):
    """
    Helper to retry Gemini API calls in case of temporary overloads (503) or rate limits (429).
    """
    delay = 1.5
    for attempt in range(max_retries):
        try:
            if config:
                return client.models.generate_content(model=model_name, contents=contents, config=config)
            else:
                return client.models.generate_content(model=model_name, contents=contents)
        except Exception as e:
            # Check if it's a transient server busy (503) or rate limit (429) error
            err_str = str(e).upper()
            is_transient = "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str or "EXHAUSTED" in err_str
            
            if is_transient and attempt < max_retries - 1:
                print(f"⚠️ Gemini API returned transient error ({e}). Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise e

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
    global client
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    event_type = data.get("event_type", "General Event")
    crowd_count = int(data.get("crowd_count", 100))
    venue_size_sqm = int(data.get("venue_size_sqm", 50))
    budget_range = data.get("budget_range", "Unknown")
    environment = data.get("environment", "Indoor")
    requirements = data.get("requirements", ["Audio", "Lighting", "Staging"])
    description = data.get("description", "")

    if not client:
        GEMINI_API_KEY_RETRY = os.getenv("GEMINI_API_KEY")
        if GEMINI_API_KEY_RETRY:
            client = genai.Client(api_key=GEMINI_API_KEY_RETRY)
        else:
            return jsonify({"error": "Gemini Client is not configured. Key missing."}), 500

    try:
        # === AGENT 1: Audio Specialist Agent (Flash) ===
        audio_response = "Not requested by user."
        if "Audio" in requirements:
            print(f"[Agent 1: Audio Specialist] Analyzing audio requirements for {crowd_count} people in {environment}...")
            audio_prompt = f"""
            You are the SoundScout Audio Specialist Agent.
            Analyze the audio requirements for this event:
            - Event Type: {event_type}
            - Crowd Count: {crowd_count} people
            - Venue Size: {venue_size_sqm} sqm
            - Environment: {environment}
            - User Event Description: {description}
            
            Recommend specific audio hardware (speakers, subwoofers, mixers, microphones) including exact quantities.
            Take into account whether it is Indoor or Outdoor (Outdoor requires more power and weather-proofing).
            Make sure to tailor the recommendations to the event description provided.
            Provide a concise, professional list and explanation of the audio choices.
            """
            audio_res = generate_content_with_retry(
                model_name='gemini-3.1-flash-lite',
                contents=audio_prompt
            )
            audio_response = audio_res.text
            print("✅ [Agent 1] Completed audio recommendation.")

        # === AGENT 2: Visual & Lighting Specialist Agent (Flash) ===
        visual_response = "Not requested by user."
        if "Lighting" in requirements or "Visuals" in requirements:
            print(f"[Agent 2: Visual Specialist] Analyzing lighting & screens for {venue_size_sqm} sqm {environment}...")
            visual_prompt = f"""
            You are the SoundScout Visual and Lighting Specialist Agent.
            Analyze the lighting and display requirements for this event:
            - Event Type: {event_type}
            - Crowd Count: {crowd_count} people
            - Venue Size: {venue_size_sqm} sqm
            - Environment: {environment}
            - User Event Description: {description}
    
            Recommend visual displays (LED walls, projectors) and lighting equipment (LED Pars, moving heads, lasers) with exact quantities.
            Take into account whether it is Indoor or Outdoor (e.g. projectors might not work well outdoors during day).
            Make sure to tailor the recommendations to the event description provided.
            Provide a concise, professional list.
            """
            visual_res = generate_content_with_retry(
                model_name='gemini-3.1-flash-lite',
                contents=visual_prompt
            )
            visual_response = visual_res.text
            print("✅ [Agent 2] Completed visual recommendation.")

        # === DETERMINISTIC POWER CALCULATOR TOOL ===
        audio_items_est = audio_response.count("\n") + 5
        visual_items_est = visual_response.count("\n") + 5
        power_calc = calculate_power_needs(audio_items_est, visual_items_est)

        # === AGENT 3: Power & Logistics Agent (Flash) ===
        logistics_response = "Not requested by user."
        if "Staging" in requirements or "Power" in requirements or len(requirements) > 0:
            print(f"[Agent 3: Logistics Specialist] Calculating power load and staging for {environment}...")
            logistics_prompt = f"""
            You are the SoundScout Power and Staging Logistics Agent.
            Review the following equipment list recommendations for an {environment} event:
            
            Audio Plan:
            {audio_response}
            
            Visual Plan:
            {visual_response}
    
            Deterministic Calculations Provided:
            - Estimated power draw: {power_calc['total_draw_kw']} kW
            - Suggested Generator: {power_calc['suggested_generator']}
            - User Event Description: {description}
    
            Formulate a staging, cabling, and power distribution plan. For outdoor events, emphasize weather protection (tents, cable ramps) and robust generators.
            """
            logistics_res = generate_content_with_retry(
                model_name='gemini-3.1-flash-lite',
                contents=logistics_prompt
            )
            logistics_response = logistics_res.text
            print("✅ [Agent 3] Completed logistics recommendation.")

        # === TOOL: Query Scikit-Learn Random Forest Model ===
        predicted_cost = predict_fair_price(crowd_count, venue_size_sqm)

        # === AGENT 4: Lead Coordinator Agent (Pro) ===
        print(f"[Agent 4: Lead Coordinator] Consolidating plan into Budget and Premium options...")
        coordinator_prompt = f"""
        You are the Lead Coordinator Agent (AV Technical Director) for SoundScout AI.
        
        You must review the specialized plans from your sub-agents:
        - Audio Specialist Plan: {audio_response}
        - Visual Specialist Plan: {visual_response}
        - Logistics Plan: {logistics_response}
        
        Constraints:
        - User's Requested Environment: {environment}
        - User's Requested Categories: {requirements}
        - User's Custom Event Description: {description}
        - User's Target Budget: LKR {budget_range}
        - ML Predicted Base Market Cost: LKR {predicted_cost:,.2f}
        
        Your Task:
        1. Consolidate these plans into TWO distinct equipment lists: a "budget_plan" and a "premium_plan".
        2. Adjust quantities and models so the "budget_plan" option stays as close to the Target Budget and ML Base Cost as possible.
        3. The "premium_plan" option should feature higher-end gear (like Line Arrays instead of standard PA, LED walls instead of simple lighting, etc) for a higher price tier.
        4. You MUST heavily customize the equipment models and brands based on the "User's Custom Event Description". For example:
           - If they request "premium sounds", "high quality audio", "vip staging", or similar high-end terms: place top-tier professional brands (e.g., L-Acoustics speakers, Shure Axient wireless, Allen & Heath SQ series digital mixers) in BOTH plans, but adjust quantities to fit the respective tiers.
           - If they request "simple system", "low budget", "within my budget", or similar low-end terms: scale down BOTH plans to standard, highly affordable models (e.g., Mackie Thump active speakers, Behringer analog mixers, wired mics).
        5. Compare the User's Target Budget Max (from Target Budget LKR {budget_range}) to the complexity of the event (Crowd size: {crowd_count}, Categories: {requirements}) and the ML Predicted Base Market Cost (LKR {predicted_cost:,.2f}).
           - If the user's budget range is not feasible to fulfill the basic requirements adequately (e.g., budget max of Rs. 20,000 for a crowd of 200+ with audio/lighting): set "feasibility_warning" to a helpful warning explaining why the budget is too low for this scope.
           - If a warning is generated, also provide "price_cutting_tips" (an array of strings showing places they can compromise to achieve the Minimum Viable Product (MVP) within or near their budget, e.g., "Use standard active PA speakers instead of passive line-arrays", "Reduce the staging size", "Skip lighting and rely on the house lights").
           - If the budget is fully feasible and safe, set "feasibility_warning" to null and "price_cutting_tips" to null.
        6. If a category (e.g. "Lighting" or "Staging") is not in the User's Requested Categories, do NOT include any equipment for that category in either of the plans.
        7. Output the result STRICTLY as a JSON object with four keys: "budget_plan", "premium_plan", "feasibility_warning", and "price_cutting_tips". Each key must contain either an array of strings representing the equipment, a string warning, or null. Do not output markdown, notes, or extra text.

        Example Output format:
        {{
            "budget_plan": [
                "2x PA Speakers (Standard)",
                "1x 6-Channel Analog Mixer",
                "2x Wired Handheld Microphones"
            ],
            "premium_plan": [
                "4x Line Array Speakers (Premium)",
                "2x Subwoofers",
                "1x 16-Channel Digital Mixer",
                "4x Wireless Shure Microphones"
            ],
            "feasibility_warning": "Your budget of LKR 20,000 is low for a crowd of 200 with both audio and lighting. Standard rentals usually start around LKR 50,000.",
            "price_cutting_tips": [
                "Opt for 2 simple PA speakers instead of adding subwoofers",
                "Reduce lighting fixtures to basic static LED Par Cans only",
                "Consider dropping the staging requirement to save up to LKR 15,000"
            ]
        }}
        """
        
        coordinator_res = generate_content_with_retry(
            model_name='gemini-3.1-flash-lite',
            contents=coordinator_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        result_json = json.loads(coordinator_res.text)
        
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
