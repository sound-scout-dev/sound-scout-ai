# ai/app.py
import os
import sys
import json
import time

# Windows' default console codepage (cp1252) can't encode the emoji used in
# this file's print() calls, which crashes the process before Flask starts.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from flask import Flask, request, jsonify
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END

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
    crowds = np.random.randint(20, 2000, size=num_samples)
    venue_sizes = crowds * np.random.uniform(0.5, 1.5, size=num_samples)
    venue_sizes = np.clip(venue_sizes, 10, 1500).astype(int)
    
    prices = []
    for crowd, size in zip(crowds, venue_sizes):
        base = 8000
        base += crowd * 70
        base += size * 120
        
        if crowd > 1000:
            base += 450000
        elif crowd > 500:
            base += 180000
        elif crowd > 150:
            base += 50000
        elif crowd > 50:
            base += 15000
            
        noise = np.random.uniform(-0.12, 0.12) * base
        price = max(10000.0, round(base + noise, -3))
        prices.append(price)
        
    X = np.column_stack((crowds, venue_sizes))
    y = np.array(prices)
    return X, y

print("⚙️ Generating historical Sri Lankan AV rental dataset...")
X_train, y_train = generate_sri_lankan_synthetic_data(250)
print(f"📊 Training Random Forest model on {len(X_train)} quotes...")
pricing_model = RandomForestRegressor(n_estimators=100, random_state=42)
pricing_model.fit(X_train, y_train)
print("✅ Random Forest pricing model trained and online.")

# Train secondary Random Forest to estimate venue size based on crowd size count
print("⚙️ Training secondary Random Forest to estimate venue size from crowd count...")
X_venue = X_train[:, 0].reshape(-1, 1) # crowd sizes
y_venue = X_train[:, 1]                # venue sizes
venue_model = RandomForestRegressor(n_estimators=100, random_state=42)
venue_model.fit(X_venue, y_venue)
print("✅ Secondary venue estimation model trained and online.")

def predict_venue_size(crowd_count: int) -> int:
    try:
        prediction = venue_model.predict([[crowd_count]])
        return int(max(10, round(prediction[0])))
    except Exception as e:
        print(f"Venue Model Error: {e}")
        return int(crowd_count * 1.0)

def predict_fair_price(crowd_count: int, venue_size_sqm: int) -> float:
    """Predicts a baseline fair market price in Sri Lankan Rupees (LKR)."""
    try:
        prediction = pricing_model.predict([[crowd_count, venue_size_sqm]])
        return float(max(10000.0, round(prediction[0], 2)))
    except Exception as e:
        print(f"Pricing Model Error: {e}")
        return 120000.0

# ----------------- LOGISTICS POWER CALCULATOR (Deterministic Tool) -----------------
def calculate_power_needs(audio_items_count: int, visual_items_count: int) -> dict:
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
def generate_content_with_retry(model_name, contents, config=None, max_retries=6):
    delay = 2.0
    for attempt in range(max_retries):
        try:
            if config:
                return client.models.generate_content(model=model_name, contents=contents, config=config)
            else:
                return client.models.generate_content(model=model_name, contents=contents)
        except Exception as e:
            err_str = str(e).upper()
            is_transient = "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str or "EXHAUSTED" in err_str
            
            if is_transient and attempt < max_retries - 1:
                sleep_time = 5.5 if "429" in err_str or "EXHAUSTED" in err_str else delay
                print(f"⚠️ Gemini API returned transient error ({e}). Retrying in {sleep_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(sleep_time)
                delay *= 2
            else:
                raise e

def resolve_district_with_ai(location):
    if not location or location.strip() == '':
        return "Colombo"
    
    prompt = f"""
    You are a Sri Lankan Location Expert.
    Identify the exact Sri Lankan district that the following venue/place belongs to:
    Venue: "{location}"
    
    Respond with ONLY the name of the district from this list:
    - Colombo
    - Gampaha
    - Kalutara
    - Kandy
    - Matale
    - Nuwara Eliya
    - Galle
    - Matara
    - Hambantota
    - Jaffna
    - Kilinochchi
    - Mannar
    - Vavuniya
    - Mullaitivu
    - Batticaloa
    - Ampara
    - Trincomalee
    - Kurunegala
    - Puttalam
    - Anuradhapura
    - Polonnaruwa
    - Badulla
    - Moneragala
    - Ratnapura
    - Kegalle
    
    If the venue is ambiguous or not in Sri Lanka, suggest the most likely district or "Colombo".
    Do not output any markdown, explanations, or extra text. Output exactly one word (the district name).
    """
    try:
        res = generate_content_with_retry(
            model_name='gemini-3.1-flash-lite',
            contents=prompt
        )
        district = res.text.strip().replace("*", "").replace('"', '').replace("'", "")
        return district
    except Exception as e:
        print(f"AI District Resolution Failed: {e}")
        return "Colombo"

# ----------------- LANGGRAPH ARCHITECTURE -----------------

class GraphState(TypedDict):
    # Scoping parameters inputs
    event_type: str
    crowd_count: int
    venue_size_sqm: int
    budget_range: str
    environment: str
    requirements: List[str]
    description: str
    location: str
    
    # Sub-agent recommendations text
    audio_recommendation: Optional[str]
    visual_recommendation: Optional[str]
    logistics_recommendation: Optional[str]
    
    # Calculations & Model predictions
    ml_predicted_cost: float
    calculated_power_draw_kw: float
    suggested_generator: str
    
    # Coordinator final plan choices
    budget_plan: List[str]
    premium_plan: List[str]
    feasibility_warning: Optional[str]
    price_cutting_tips: List[str]
    resolved_district: str
    
    # Loop feedback control variables
    scaling_instruction: Optional[str]
    loop_count: int

# Node 1: Audio Specialist Agent
def audio_node(state: GraphState) -> dict:
    if "Audio" not in state["requirements"]:
        return {"audio_recommendation": "Not requested by user."}
    
    venue_size = state.get("venue_size_sqm")
    if venue_size is None or venue_size <= 0:
        venue_size = predict_venue_size(state.get("crowd_count", 100))

    print(f"🎙️ [LangGraph Node: Audio] Analyzing crowd of {state['crowd_count']} at {state['location']} (using venue size {venue_size} sqm)...")
    scaling_prompt = ""
    if state.get("scaling_instruction"):
        scaling_prompt = f"\nCRITICAL BUDGET CONSTRAINT FOR SCALING DOWN: {state['scaling_instruction']}"

    prompt = f"""
    You are the SoundScout Audio Specialist Agent.
    Analyze the audio requirements for this event:
    - Event Type: {state['event_type']}
    - Crowd Count: {state['crowd_count']} people
    - Venue Size: {venue_size} sqm
    - Environment: {state['environment']}
    - Location/Venue: {state['location']}
    - User Event Description: {state['description']}
    {scaling_prompt}
    
    Recommend specific audio hardware (speakers, subwoofers, mixers, microphones) including exact quantities.
    Take into account whether it is Indoor or Outdoor (Outdoor requires more power and weather-proofing).
    Make sure to tailor the recommendations to the event description and any budget constraints.
    Provide a concise, professional list and explanation of the audio choices.
    """
    res = generate_content_with_retry(
        model_name='gemini-3.1-flash-lite',
        contents=prompt
    )
    return {"audio_recommendation": res.text}

# Node 2: Visual Specialist Agent
def visual_node(state: GraphState) -> dict:
    if "Lighting" not in state["requirements"] and "Visuals" not in state["requirements"]:
        return {"visual_recommendation": "Not requested by user."}
    
    venue_size = state.get("venue_size_sqm")
    if venue_size is None or venue_size <= 0:
        venue_size = predict_venue_size(state.get("crowd_count", 100))

    print(f"💡 [LangGraph Node: Visual] Analyzing screen/light needs for venue size {venue_size} sqm...")
    scaling_prompt = ""
    if state.get("scaling_instruction"):
        scaling_prompt = f"\nCRITICAL BUDGET CONSTRAINT FOR SCALING DOWN: {state['scaling_instruction']}"

    prompt = f"""
    You are the SoundScout Visual and Lighting Specialist Agent.
    Analyze the lighting and display requirements for this event:
    - Event Type: {state['event_type']}
    - Crowd Count: {state['crowd_count']} people
    - Venue Size: {venue_size} sqm
    - Environment: {state['environment']}
    - Location/Venue: {state['location']}
    - User Event Description: {state['description']}
    {scaling_prompt}
    
    Recommend visual displays (LED walls, projectors) and lighting equipment (LED Pars, moving heads, lasers) with exact quantities.
    Take into account whether it is Indoor or Outdoor (e.g. projectors might not work well outdoors during day).
    Make sure to tailor the recommendations to the event description.
    Provide a concise, professional list.
    """
    res = generate_content_with_retry(
        model_name='gemini-3.1-flash-lite',
        contents=prompt
    )
    return {"visual_recommendation": res.text}

# Node 3: Staging & Power Logistics Agent
def logistics_node(state: GraphState) -> dict:
    has_staging = "Staging" in state["requirements"]
    has_power = "Power" in state["requirements"]
    if not has_staging and not has_power and len(state["requirements"]) == 0:
        return {
            "logistics_recommendation": "Not requested by user.",
            "calculated_power_draw_kw": 0.0,
            "suggested_generator": "None"
        }
    
    print(f"📦 [LangGraph Node: Logistics] Calculating deterministic load distribution...")
    audio_items = state["audio_recommendation"].count("\n") + 5
    visual_items = state["visual_recommendation"].count("\n") + 5
    power_calc = calculate_power_needs(audio_items, visual_items)

    prompt = f"""
    You are the SoundScout Power and Staging Logistics Agent.
    Review the following equipment list recommendations for an {state['environment']} event at {state['location']}:
    
    Audio Plan:
    {state['audio_recommendation']}
    
    Visual Plan:
    {state['visual_recommendation']}
    
    Deterministic Calculations Provided:
    - Estimated power draw: {power_calc['total_draw_kw']} kW
    - Suggested Generator: {power_calc['suggested_generator']}
    - Location/Venue: {state['location']}
    - User Event Description: {state['description']}
    
    Formulate a staging, cabling, and power distribution plan. For outdoor events, emphasize weather protection (tents, cable ramps) and robust generators.
    """
    res = generate_content_with_retry(
        model_name='gemini-3.1-flash-lite',
        contents=prompt
    )
    return {
        "logistics_recommendation": res.text,
        "calculated_power_draw_kw": power_calc["total_draw_kw"],
        "suggested_generator": power_calc["suggested_generator"]
    }

# Node 4: Lead Coordinator Node (Budget evaluation logic)
def coordinator_node(state: GraphState) -> dict:
    print(f"👑 [LangGraph Node: Coordinator] Consolidating and comparing limits...")
    crowd = max(1, state.get("crowd_count", 100))
    venue_size = state.get("venue_size_sqm")
    if venue_size is None or venue_size <= 0:
        venue_size = predict_venue_size(crowd)
    predicted_cost = predict_fair_price(crowd, venue_size)

    prompt = f"""
    You are the Lead Coordinator Agent (AV Technical Director) for SoundScout AI.
    
    You must review the specialized plans from your sub-agents:
    - Audio Specialist Plan: {state['audio_recommendation']}
    - Visual Specialist Plan: {state['visual_recommendation']}
    - Logistics Plan: {state['logistics_recommendation']}
    
    Constraints:
    - User's Requested Environment: {state['environment']}
    - User's Requested Location/Venue: {state['location']}
    - User's Requested Categories: {state['requirements']}
    - User's Custom Event Description: {state['description']}
    - User's Target Budget: LKR {state['budget_range']}
    - ML Predicted Base Market Cost: LKR {predicted_cost:,.2f}
    
    Your Task:
    1. Consolidate these plans into TWO distinct equipment lists: a "budget_plan" and a "premium_plan".
    2. Adjust quantities and models so the "budget_plan" option stays as close to the Target Budget and ML Base Cost as possible. If the budget is extremely tight, scale down only the "budget_plan" to standard models.
    3. The "premium_plan" option MUST ALWAYS feature high-end, premium gear (such as Line Arrays, professional moving heads, large generators) and larger quantities, REGARDLESS of how tight the user's budget is. Do NOT scale down the "premium_plan" to fit the low budget; it should remain a high-quality showcase option for better performance.
    4. You MUST heavily customize the equipment models and brands based on the "User's Custom Event Description".
    5. Compare the User's Target Budget Max (from Target Budget LKR {state['budget_range']}) to the estimated cost of the budget plan.
       - Note that the budget plan's estimated price range is calculated and shown to the user as LKR {predicted_cost * 0.8:,.0f} to LKR {predicted_cost * 1.2:,.0f}.
       - You MUST ONLY generate a feasibility warning if the budget plan's high limit (LKR {predicted_cost * 1.2:,.0f}) exceeds the User's Target Budget Max limit.
       - If it does exceed, set "feasibility_warning" to a helpful warning explaining why the budget is too low for this scope. In "price_cutting_tips", suggest items they can remove/compromise on to achieve the MVP (e.g., "Utilize standard active PA speakers instead of passive line-arrays").
       - If it does NOT exceed, set "feasibility_warning" to null. However, if the user's budget max is significantly higher than the premium plan's high limit (representing a highly surplus/over-sufficient budget), you can use "price_cutting_tips" to suggest premium upgrades or redundant gear they could drop (e.g., 'You have surplus budget. Suggest adding side delay speaker stacks for improved coverage' or 'Suggest upgrading standard monitors to professional in-ear monitors').
    6. For any high-end, luxury, or non-essential equipment items, append `(Optional: <Brief explanation comment why this is not compulsory for a basic setup>)` directly to the equipment item string.
    7. If a category is not in the User's Requested Categories, do NOT include any equipment for that category.
    8. Output the result STRICTLY as a JSON object with four keys: "budget_plan", "premium_plan", "feasibility_warning", and "price_cutting_tips". Do not output markdown, notes, or extra text.
    """
    res = generate_content_with_retry(
        model_name='gemini-3.1-flash-lite',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    result = json.loads(res.text)

    # Budget matching logic
    target_max = 999999999.0
    if state["budget_range"] and "-" in state["budget_range"]:
        try:
            target_max = float(state["budget_range"].split("-")[1].replace(",", "").strip())
        except Exception:
            pass

    budget_high = predicted_cost * 1.2
    scaling_instruction = None
    loop_count = state.get("loop_count", 0)

    # If the budget plan's maximum estimated price exceeds the user's budget ceiling
    if budget_high > target_max:
        if loop_count < 1:
            print(f"⚠️ Budget high {budget_high} exceeds max {target_max}. Scaling down once...")
            scaling_instruction = (
                f"The calculated cost LKR {budget_high:,.0f} exceeds the user's budget max LKR {target_max:,.0f}. "
                "Re-spec the equipment: reduce counts, choose standard active speakers, decrease lighting, and simplify stage dimensions."
            )
            loop_count += 1
        else:
            print(f"⚠️ After scaling down once, budget plan high {budget_high} is still over budget. Raising warnings...")
            result["feasibility_warning"] = (
                f"Your budget max of LKR {target_max:,.0f} is insufficient for a crowd of {state['crowd_count']} "
                f"with {', '.join(state['requirements'])}. The budget plan shown has been scaled to the absolute minimum viable setup, "
                "but it may not be fully sufficient for the requested scale."
            )

    resolved_district = resolve_district_with_ai(state["location"])

    return {
        "budget_plan": result.get("budget_plan", []),
        "premium_plan": result.get("premium_plan", []),
        "feasibility_warning": result.get("feasibility_warning"),
        "price_cutting_tips": result.get("price_cutting_tips", []),
        "resolved_district": resolved_district,
        "ml_predicted_cost": predicted_cost,
        "scaling_instruction": scaling_instruction,
        "loop_count": loop_count
    }

# Build and compile the LangGraph workflow
workflow = StateGraph(GraphState)
workflow.add_node("audio", audio_node)
workflow.add_node("visual", visual_node)
workflow.add_node("logistics", logistics_node)
workflow.add_node("coordinator", coordinator_node)

workflow.add_edge(START, "audio")
workflow.add_edge("audio", "visual")
workflow.add_edge("visual", "logistics")
workflow.add_edge("logistics", "coordinator")

def route_coordinator(state: GraphState):
    if state.get("scaling_instruction") and state.get("loop_count", 0) <= 1:
        print("🔄 Loops detected. Routing back to Audio Node for scaling...")
        return "loop_back"
    return "finish"

workflow.add_conditional_edges(
    "coordinator",
    route_coordinator,
    {
        "loop_back": "audio",
        "finish": END
    }
)

app_graph = workflow.compile()

# ----------------- FLASK ENDPOINTS -----------------

@app.route('/api/generate', methods=['POST'])
def generate_infrastructure_plan():
    global client
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    event_type = data.get("event_type", "General Event")
    crowd_count = int(data.get("crowd_count", 100))
    
    # Optional venue size handling: if empty or <= 0, predict size using Random Forest model
    venue_size_sqm = data.get("venue_size_sqm")
    if venue_size_sqm is None or str(venue_size_sqm).strip() == "":
        venue_size_sqm = predict_venue_size(crowd_count)
    else:
        try:
            venue_size_sqm = int(venue_size_sqm)
            if venue_size_sqm <= 0:
                venue_size_sqm = predict_venue_size(crowd_count)
        except ValueError:
            venue_size_sqm = predict_venue_size(crowd_count)

    budget_range = data.get("budget_range", "Unknown")
    environment = data.get("environment", "Indoor")
    requirements = data.get("requirements", ["Audio", "Lighting", "Staging"])
    description = data.get("description", "")
    location = data.get("location", "")

    if not client:
        GEMINI_API_KEY_RETRY = os.getenv("GEMINI_API_KEY")
        if GEMINI_API_KEY_RETRY:
            client = genai.Client(api_key=GEMINI_API_KEY_RETRY)
        else:
            return jsonify({"error": "Gemini Client is not configured. Key missing."}), 500

    try:
        initial_state = {
            "event_type": event_type,
            "crowd_count": crowd_count,
            "venue_size_sqm": venue_size_sqm,
            "budget_range": budget_range,
            "environment": environment,
            "requirements": requirements,
            "description": description,
            "location": location,
            
            "audio_recommendation": None,
            "visual_recommendation": None,
            "logistics_recommendation": None,
            
            "ml_predicted_cost": 0.0,
            "calculated_power_draw_kw": 0.0,
            "suggested_generator": "",
            
            "budget_plan": [],
            "premium_plan": [],
            "feasibility_warning": None,
            "price_cutting_tips": [],
            "resolved_district": "",
            
            "scaling_instruction": None,
            "loop_count": 0
        }
        
        # Execute the compiled multi-agent state graph
        print("🚀 [LangGraph Pipeline] Starting graph execution...")
        final_output = app_graph.invoke(initial_state)
        print("🎉 [LangGraph Pipeline] Graph execution complete.")

        response_payload = {
            "budget_plan": final_output["budget_plan"],
            "premium_plan": final_output["premium_plan"],
            "feasibility_warning": final_output["feasibility_warning"],
            "price_cutting_tips": final_output["price_cutting_tips"],
            "resolved_district": final_output["resolved_district"],
            "ml_predicted_cost": final_output["ml_predicted_cost"],
            "scoped_budget_range": final_output["budget_range"],
            "calculated_power_draw_kw": final_output["calculated_power_draw_kw"],
            "agent_spec_log": {
                "audio_summary": (final_output["audio_recommendation"] or "")[:300] + "...",
                "visual_summary": (final_output["visual_recommendation"] or "")[:300] + "...",
                "logistics_summary": (final_output["logistics_recommendation"] or "")[:300] + "..."
            }
        }
        return jsonify(response_payload), 200
    except Exception as e:
        print(f"Graph Execution Failed: {e}")
        return jsonify({"error": f"Internal AI Graph error: {str(e)}"}), 500

@app.route('/api/distance', methods=['POST'])
def estimate_distance():
    data = request.get_json() or {}
    venue = data.get("venue", "")
    vendor_region = data.get("vendor_region", "")
    
    if not venue or not vendor_region:
        return jsonify({"distance_km": 15.0}), 200
        
    prompt = f"""
    You are a Sri Lankan Geography Expert.
    Estimate the rough road distance in kilometers (km) between:
    Event Venue: "{venue}"
    Vendor Base/Region: "{vendor_region}"
    
    Respond with ONLY a number representing the estimated distance in km (e.g., 12.5 or 118).
    Do not output any markdown, explanations, or units. Just a single float or integer.
    """
    try:
        res = generate_content_with_retry('gemini-3.1-flash-lite', prompt)
        dist_str = res.text.strip().replace("km", "").replace("*", "").replace('"', '').replace("'", "")
        dist = float(dist_str)
        return jsonify({"distance_km": dist}), 200
    except Exception as e:
        print(f"Error estimating distance: {e}")
        return jsonify({"distance_km": 25.0}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
