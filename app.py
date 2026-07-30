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

# Configure Gemini Client pool (supports multiple comma-separated keys for automatic failover)
raw_keys = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]
current_key_idx = 0

def get_genai_client():
    global current_key_idx
    if not GEMINI_API_KEYS:
        return None
    key = GEMINI_API_KEYS[current_key_idx % len(GEMINI_API_KEYS)]
    try:
        return genai.Client(api_key=key)
    except Exception as e:
        print(f"Error initializing GenAI Client for key index #{current_key_idx}: {e}")
        return None

def rotate_genai_key():
    global current_key_idx
    if len(GEMINI_API_KEYS) > 1:
        current_key_idx = (current_key_idx + 1) % len(GEMINI_API_KEYS)
        print(f"🔄 Rotated to Gemini API key index #{current_key_idx}")

client = get_genai_client()
if not GEMINI_API_KEYS:
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

# ----------------- RETRY HELPER FOR TRANSIENT API ERRORS & ULTRA-LOW COST MODEL FALLBACK -----------------
# Optimized for minimum token cost ($0.0375 - $0.075 per million tokens)
FALLBACK_MODELS = ['gemini-2.0-flash-lite', 'gemini-1.5-flash-8b', 'gemini-2.0-flash']

# ----------------- GROQ FREE-TIER FALLBACK PROVIDER -----------------
class DummyResponse:
    def __init__(self, text):
        self.text = text

def call_groq_api(contents):
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return None
    try:
        import requests
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        prompt_text = ""
        if isinstance(contents, str):
            prompt_text = contents
        elif isinstance(contents, list):
            extracted = []
            for item in contents:
                if isinstance(item, str):
                    extracted.append(item)
                elif hasattr(item, 'parts') and item.parts:
                    for part in item.parts:
                        if hasattr(part, 'text') and part.text:
                            extracted.append(part.text)
                elif hasattr(item, 'text') and item.text:
                    extracted.append(item.text)
            prompt_text = "\n".join(extracted)
        
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0.7
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            reply = data["choices"][0]["message"]["content"]
            print("⚡ Successfully processed request via Groq AI Fallback!")
            return DummyResponse(reply)
        else:
            print(f"⚠️ Groq API Error ({resp.status_code}): {resp.text}")
            return None
    except Exception as e:
        print(f"⚠️ Groq Fallback Exception: {e}")
        return None

def generate_content_with_retry(model_name, contents, config=None, max_retries=3):
    models_to_try = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]
    last_exception = None

    for target_model in models_to_try:
        delay = 1.5
        for attempt in range(max_retries):
            try:
                active_client = get_genai_client()
                if not active_client:
                    raise Exception("No active GEMINI_API_KEY available.")
                    
                if config:
                    return active_client.models.generate_content(model=target_model, contents=contents, config=config)
                else:
                    return active_client.models.generate_content(model=target_model, contents=contents)
            except Exception as e:
                err_str = str(e).upper()
                last_exception = e
                is_quota = "429" in err_str or "EXHAUSTED" in err_str or "RESOURCE_EXHAUSTED" in err_str
                is_transient = "503" in err_str or "UNAVAILABLE" in err_str
                
                if is_quota:
                    print(f"⚠️ Model '{target_model}' quota exhausted ({e}). Rotating key & trying fallback model...")
                    rotate_genai_key()
                    break  # Try next model & key combination immediately
                elif is_transient and attempt < max_retries - 1:
                    print(f"⚠️ Transient error on '{target_model}'. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    break

    # If all Gemini models fail or return limit:0 quota errors, attempt Groq fallback
    if os.getenv("GROQ_API_KEY"):
        print("🔄 All Gemini models failed due to free-tier quota limits. Falling back to Groq API...")
        groq_res = call_groq_api(contents)
        if groq_res:
            return groq_res

    if last_exception:
        raise last_exception
    else:
        raise Exception("All Gemini fallback models and key rotation attempts failed.")

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
            model_name='gemini-2.0-flash-lite',
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
    venue_photo_analysis: Optional[dict]
    
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

    photo_analysis_prompt = ""
    if state.get("venue_photo_analysis"):
        analysis = state["venue_photo_analysis"]
        insights = analysis.get("visual_insights", [])
        insights_str = ", ".join(insights) if insights else "None"
        photo_analysis_prompt = f"""
    - Venue Spatial/Acoustic Analysis (from Uploaded Image):
      * Reflective Surfaces (concrete/glass echo): {analysis.get('reflective_surfaces', False)}
      * Low Ceiling (<4m): {analysis.get('low_ceiling', False)}
      * Outdoor Audio Dissipation: {analysis.get('outdoor_dissipation', False)}
      * Visual Insights & Warnings: {insights_str}
        """

    prompt = f"""
    You are the SoundScout Audio Specialist Agent.
    Analyze the audio requirements for this event:
    - Event Type: {state['event_type']}
    - Crowd Count: {state['crowd_count']} people
    - Venue Size: {venue_size} sqm
    - Environment: {state['environment']}
    - Location/Venue: {state['location']}
    - User Event Description: {state['description']}
    {photo_analysis_prompt}
    {scaling_prompt}
    
    Recommend specific audio hardware (speakers, subwoofers, mixers, microphones) including exact quantities.
    Take into account whether it is Indoor or Outdoor (Outdoor requires more power and weather-proofing).
    Adjust speaker direction/quantity if Reflective Surfaces is True (e.g. recommend directional speakers or low-frequency limits to prevent concrete echo).
    Make sure to tailor the recommendations to the event description and any budget constraints.
    Provide a concise, professional list and explanation of the audio choices.
    """
    res = generate_content_with_retry(
        model_name='gemini-2.0-flash-lite',
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

    photo_analysis_prompt = ""
    if state.get("venue_photo_analysis"):
        analysis = state["venue_photo_analysis"]
        insights = analysis.get("visual_insights", [])
        insights_str = ", ".join(insights) if insights else "None"
        photo_analysis_prompt = f"""
    - Venue Spatial/Ambient Light Analysis (from Uploaded Image):
      * High Ambient Light (washes out projectors): {analysis.get('high_ambient_light', False)}
      * Low Ceiling (<4m limits tall rigging): {analysis.get('low_ceiling', False)}
      * Visual Insights & Warnings: {insights_str}
        """

    prompt = f"""
    You are the SoundScout Visual and Lighting Specialist Agent.
    Analyze the lighting and display requirements for this event:
    - Event Type: {state['event_type']}
    - Crowd Count: {state['crowd_count']} people
    - Venue Size: {venue_size} sqm
    - Environment: {state['environment']}
    - Location/Venue: {state['location']}
    - User Event Description: {state['description']}
    {photo_analysis_prompt}
    {scaling_prompt}
    
    Recommend visual displays (LED walls, projectors) and lighting equipment (LED Pars, moving heads, lasers) with exact quantities.
    Take into account whether it is Indoor or Outdoor.
    If High Ambient Light is True, heavily prefer high-nit LED walls over projectors.
    If Low Ceiling is True, avoid tall vertical trusses or massive fixtures, and specify compact ground-stacked or T-bar light options.
    Make sure to tailor the recommendations to the event description.
    Provide a concise, professional list.
    """
    res = generate_content_with_retry(
        model_name='gemini-2.0-flash-lite',
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
        model_name='gemini-2.0-flash-lite',
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
        model_name='gemini-2.0-flash-lite',
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

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "SoundScout AI Service is running! 🚀"}), 200

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
    venue_photo_analysis = data.get("venue_photo_analysis", None)

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
            "venue_photo_analysis": venue_photo_analysis,
            
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
        res = generate_content_with_retry('gemini-2.0-flash-lite', prompt)
        dist_str = res.text.strip().replace("km", "").replace("*", "").replace('"', '').replace("'", "")
        dist = float(dist_str)
        return jsonify({"distance_km": dist}), 200
    except Exception as e:
        print(f"Error estimating distance: {e}")
        return jsonify({"distance_km": 25.0}), 200

@app.route('/api/voice-intake', methods=['POST'])
def process_voice_intake():
    global client
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
        
    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    
    if not client:
        GEMINI_API_KEY_RETRY = os.getenv("GEMINI_API_KEY")
        if GEMINI_API_KEY_RETRY:
            client = genai.Client(api_key=GEMINI_API_KEY_RETRY)
        else:
            return jsonify({"error": "Gemini Client is not configured. Key missing."}), 500

    prompt = """
    You are the SoundScout Voice Intake Agent.
    Listen to this audio note carefully. The user is describing an event they want to plan.
    It might be spoken in English, Sinhala, or a mix of both (Singlish/code-switched language).
    
    Your goal is to parse and extract the following parameters with extremely high accuracy:
    1. eventName: A suitable name for the event (e.g. "Beach Music Fest", "Sakura"). Pay close attention to Sinhala words like "Sakura" (do not transcribe as "Sakura giri" unless they explicitly say "giri").
    2. eventType: The type of the event. MUST be one of these exact values: "Music Festival", "Corporate Conference", "Wedding", "Private Party", "University Seminar", "Other".
    3. crowdSize: The expected guest count (integer, e.g. 300).
    4. venueSizeSqm: The venue size in square meters if mentioned, otherwise null.
    5. budgetMin: The minimum budget in LKR (integer).
    6. budgetMax: The maximum budget in LKR (integer).
       - Note: If they say "Five Lakhs" or "Lakhs paha", that is 500,000 LKR. 
       - If they say "One Lakh", that is 100,000 LKR.
       - Do not confuse LKR 500,000 (five lakhs) with 105,000.
    7. location: The event venue/location (string, e.g. "University of Moratuwa", "Waters Edge, Battaramulla").
    8. environment: "Indoor" or "Outdoor" if specified, otherwise default to "Indoor".
    9. requirements: A list of requested equipment categories. Must only contain values from: ["Audio", "Lighting", "Staging", "Visuals", "Power"].
    10. description: A brief, professional description of the event (minimum 25 words) based on their voice input, summarising the theme, location, and requirements.

    Respond STRICTLY with a JSON object containing a "parameters" key, whose value is the extracted parameters object.
    
    Example response shape:
    {
       "parameters": {
          "eventName": "Sakura",
          "eventType": "Music Festival",
          "crowdSize": 500,
          "venueSizeSqm": 400,
          "budgetMin": 400000,
          "budgetMax": 500000,
          "location": "University of Moratuwa",
          "environment": "Outdoor",
          "requirements": ["Audio", "Lighting", "Visuals"],
          "description": "An outdoor music festival named Sakura at the University of Moratuwa with an expected crowd of 500 people, requiring audio, lighting, and visuals reinforcement."
       }
    }
    """
    
    try:
        contents = [
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type="audio/webm"
            ),
            prompt
        ]
        
        # Using gemini-2.0-flash-lite for lightweight audio comprehension
        res = generate_content_with_retry(
            model_name='gemini-2.0-flash-lite',
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        result_json = json.loads(res.text)
        return jsonify(result_json), 200
    except Exception as e:
        print(f"Error processing voice intake: {e}")
        return jsonify({"error": f"Failed to analyze voice note: {str(e)}"}), 500

@app.route('/api/venue-analysis', methods=['POST'])
def analyze_venue():
    global client
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    image_file = request.files['image']
    image_bytes = image_file.read()
    content_type = image_file.content_type or "image/jpeg"
    
    if not client:
        GEMINI_API_KEY_RETRY = os.getenv("GEMINI_API_KEY")
        if GEMINI_API_KEY_RETRY:
            client = genai.Client(api_key=GEMINI_API_KEY_RETRY)
        else:
            return jsonify({"error": "Gemini Client is not configured. Key missing."}), 500

    prompt = """
    You are the SoundScout Venue Spatial & Acoustic Analysis Agent.
    Analyze the uploaded photo of this event venue and extract spatial and acoustic characteristics.
    
    Look for:
    1. Reflective surfaces (e.g. glass walls, concrete floors/walls, mirrors) which cause echoes.
    2. Ceiling height (e.g. low ceiling limits high staging, high ceiling requires powerful wash lighting).
    3. Ambient lighting (e.g. high ambient daylight / glass pavilions wash out projectors, dark halls don't).
    4. Space environment (e.g. outdoor open field means sound dissipation, indoor tight room causes bass build-up).
    
    Your output must strictly be a JSON object with these keys:
    - "reflective_surfaces": boolean (true if concrete/glass walls/mirrors are prominent)
    - "high_ambient_light": boolean (true if highly lit by daylight or bright overheads)
    - "low_ceiling": boolean (true if ceiling looks under 4 meters)
    - "outdoor_dissipation": boolean (true if outdoor grass or open spaces)
    - "visual_insights": list of strings (each a concise, human-readable warning or recommendation, e.g. "Hard concrete walls: add acoustic panels or space speakers", "High ambient daylight: use high-nit LED wall instead of projector").

    Respond ONLY with the raw JSON object. Do not include markdown, backticks, or other text.
    """
    
    try:
        contents = [
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=content_type
            ),
            prompt
        ]
        
        res = generate_content_with_retry(
            model_name='gemini-2.0-flash-lite',
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        result_json = json.loads(res.text)
        return jsonify(result_json), 200
    except Exception as e:
        print(f"Error analyzing venue image: {e}")
        return jsonify({"error": f"Failed to analyze venue photo: {str(e)}"}), 500

support_chats = {}

@app.route('/api/support', methods=['POST'])
def support_bot():
    global client
    data = request.get_json() or {}
    session_id = data.get("session_id")
    user_message = data.get("message")
    
    if not session_id or not user_message:
        return jsonify({"error": "session_id and message are required"}), 400
        
    if not client:
        GEMINI_API_KEY_RETRY = os.getenv("GEMINI_API_KEY")
        if GEMINI_API_KEY_RETRY:
            client = genai.Client(api_key=GEMINI_API_KEY_RETRY)
        else:
            return jsonify({"error": "Gemini Client is not configured. Key missing."}), 500
            
    system_instruction = """
    You are the SoundScout AI Support Assistant, an official WhatsApp bot helping event organizers and AV equipment vendors.
    SoundScout is a smart platform matching event organizers with audio, lighting, visuals, and staging vendors in Sri Lanka.
    Organizers can plan events using native voice notes, upload venue photos for automatic acoustic analysis, receive optimized equipment recommendations, and accept bids from vendors.
    Vendors can view open opportunities matching their categories and districts, submit bids, and register inventory.
    
    Be helpful, extremely professional, concise (since this is on WhatsApp, keep responses to maximum 3-4 bullet points or short paragraphs), and friendly.
    If asked about system status, everything is fully operational.
    """
    
    # Initialize history list if not present
    if session_id not in support_chats:
        support_chats[session_id] = []
        
    # Append the new user message to the session history
    support_chats[session_id].append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
    )
    
    # Keep history bounded to last 20 messages to avoid context overflow and memory bloat
    if len(support_chats[session_id]) > 20:
        support_chats[session_id] = support_chats[session_id][-20:]

    try:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
            max_output_tokens=300
        )
        # Use gemini-2.0-flash-lite — lightweight ultra-low cost model
        res = generate_content_with_retry(
            model_name='gemini-2.0-flash-lite',
            contents=support_chats[session_id],
            config=config
        )

        reply_text = res.text.strip()
        print(f"Support bot replied to {session_id}: {reply_text[:80]}...")

        # Append assistant reply to the history
        support_chats[session_id].append(
            types.Content(role="model", parts=[types.Part.from_text(text=reply_text)])
        )

        return jsonify({"reply": reply_text}), 200

    except Exception as e:
        print(f"Error in support agent: {e}")
        return jsonify({"error": f"Support agent error: {str(e)}"}), 500

# ── Keep-alive thread: ping ourselves every 10 minutes so Render free tier
#    doesn't shut us down between WhatsApp messages ──────────────────────────
import threading
import urllib.request

def _keep_alive():
    import time
    own_url = os.getenv("RENDER_EXTERNAL_URL", "")
    if not own_url:
        return  # not running on Render, skip
    while True:
        time.sleep(600)  # ping every 10 minutes
        try:
            urllib.request.urlopen(own_url + "/", timeout=10)
            print("🔄 Keep-alive ping sent")
        except Exception as e:
            print(f"⚠️  Keep-alive ping failed: {e}")

_ka_thread = threading.Thread(target=_keep_alive, daemon=True)
_ka_thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
