from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env from backend folder
load_dotenv(".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

# Load ML files
model = joblib.load("eco_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Create FastAPI app
app = FastAPI(title="AI Eco-Friendly Shopping Assistant API")

# Enable CORS so Lovable frontend can call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Input Schema
# ------------------------------------------------------------

class ProductInput(BaseModel):
    category: str
    price: float
    packaging_type: str
    recyclable: str
    organic: str
    durability: str
    carbon_impact: str


def normalize_input(data):
    normalized = data.copy()

    category_map = {
        "Food": "Grocery",
        "food": "Grocery",

        "Beverage": "Beverages",
        "beverage": "Beverages",
        "beverages": "Beverages",

        "Clothing": "Clothing",
        "clothing": "Clothing",

        "Electronics": "Household",
        "electronics": "Household",

        "Household": "Household",
        "household": "Household",

        "Beauty": "Personal Care",
        "beauty": "Personal Care"
    }

    normalized["category"] = category_map.get(
        str(normalized["category"]).strip(),
        "Grocery"
    )

    for col in ["packaging_type", "recyclable", "organic", "durability", "carbon_impact"]:
        normalized[col] = str(normalized[col]).strip().lower()

    return normalized

# ------------------------------------------------------------
# Encode Input for ML Model
# ------------------------------------------------------------

def encode_input(data):
    data = normalize_input(data)
    encoded = []

    for col, value in data.items():
        if col in label_encoders:
            le = label_encoders[col]

            if value not in le.classes_:
                value = le.classes_[0]

            encoded.append(le.transform([value])[0])
        else:
            encoded.append(value)

    return encoded

def format_output(data):
    formatted = data.copy()

    for key, value in formatted.items():
        if isinstance(value, str):
            formatted[key] = value.replace("_", " ").title()

    return formatted

# ------------------------------------------------------------
# Gemini Response Function
# ------------------------------------------------------------

def get_gemini_response(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            top_p=0.9,
            max_output_tokens=1500
        )
    )

    return response.text

# ------------------------------------------------------------
# Friendly System Prompt for Domain Control
# ------------------------------------------------------------

SYSTEM_PROMPT = """
You are a friendly AI Eco-Friendly Shopping Assistant.

Your job is to help users make better shopping choices for the environment.
Speak like a helpful, polite, and warm chatbot.

Your domain includes:
- eco-friendly products
- sustainable shopping
- recycling
- product alternatives
- packaging impact
- carbon impact
- responsible consumer choices

If the user asks something outside this domain:
Say sorry in a friendly way and explain that you are designed only for eco-friendly shopping guidance.
Do not answer unrelated topics.

Example:
"Sorry, I’m not the right assistant for that topic. I’m designed to help with eco-friendly shopping, sustainable products, recycling, and environmental impact."

Tone rules:
1. Be friendly, warm, and natural.
2. Avoid sounding robotic or overly technical.
3. Use simple student-friendly language.
4. Give practical and useful suggestions.
5. Do not make fake brand-specific claims.
6. Keep answers structured but conversational.
7. Use short headings only when useful.


"""

# ------------------------------------------------------------
# Home Endpoint
# ------------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "AI Eco-Friendly Shopping Assistant API is running successfully."
    }

# ------------------------------------------------------------
# Chat Endpoint
# Main chatbot feature
# ------------------------------------------------------------

@app.post("/chat")
def chat(user_query: str):

    full_prompt = f"""
{SYSTEM_PROMPT}

User question:
{user_query}

Start with a friendly natural line.

Then format the response STRICTLY like this:

Answer:
Write a short paragraph.

Why it matters:
Write a short explanation paragraph.

Better choices:
- Use bullet points ONLY for options
- Each point on a new line

Quick eco tip:
- Give 2–3 short bullet points

IMPORTANT:
- Do NOT use bullets for headings
- Headings must be plain text (no symbols)
- Always leave a blank line after each heading
- Use clean spacing for readability

Make it friendly, complete, and easy to understand.
"""

    reply = get_gemini_response(full_prompt)

    return {
        "user_query": user_query,
        "response": reply
    }

# ------------------------------------------------------------
# ML Prediction Endpoint
# Predicts eco category: Low / Medium / High
# ------------------------------------------------------------

@app.post("/predict")
def predict(input_data: ProductInput):

    data_dict = normalize_input(input_data.model_dump())

    encoded_input = encode_input(data_dict)
    prediction = model.predict([encoded_input])[0]
    eco_category = target_encoder.inverse_transform([prediction])[0]

    prediction_proba = model.predict_proba([encoded_input])[0]
    confidence = round(float(max(prediction_proba)) * 100, 2)

    if eco_category == "High":
        meaning = "This product looks like a strong sustainable choice."
        recommendation = "You can prefer this type of product because it has better eco-friendly characteristics."
    elif eco_category == "Medium":
        meaning = "This product has moderate sustainability, but there is room for improvement."
        recommendation = "Try choosing products with recyclable packaging, lower carbon impact, or better durability."
    else:
        meaning = "This product has low sustainability, so choosing a greener alternative would be better."
        recommendation = "Prefer products with recyclable packaging, low carbon impact, higher durability, or organic materials."

    return {
        "input_product": data_dict,
        "eco_category": eco_category,
        "confidence_percentage": confidence,
        "meaning": meaning,
        "recommendation": recommendation
    }

# ------------------------------------------------------------
# Smart Assistant Endpoint
# Combines ML prediction + Gemini explanation
# ------------------------------------------------------------

@app.post("/smart-assistant")
def smart_assistant(input_data: ProductInput):

    data_dict = normalize_input(input_data.model_dump())

    encoded_input = encode_input(data_dict)
    prediction = model.predict([encoded_input])[0]
    eco_category = target_encoder.inverse_transform([prediction])[0]

    prediction_proba = model.predict_proba([encoded_input])[0]
    confidence = round(float(max(prediction_proba)) * 100, 2)

    prompt = f"""
{SYSTEM_PROMPT}

You are analyzing a product for eco-friendliness.

Product details:
- Category: {data_dict['category']}
- Price: {data_dict['price']}
- Packaging: {data_dict['packaging_type']}
- Recyclable: {data_dict['recyclable']}
- Organic: {data_dict['organic']}
- Durability: {data_dict['durability']}
- Carbon impact: {data_dict['carbon_impact']}

Machine Learning predicted sustainability category: {eco_category}
Prediction confidence: {confidence}%

Give a short, to-the-point response in this exact format:

Prediction:
Write one sentence only.

Reason:
Write 2 short sentences only.

Better choices:
- Give 2 short bullet points only.

Shopping tip:
- Give 1 short bullet point only.

Do not write long paragraphs.
Do not add extra sections.

Make it friendly, complete, and easy to understand.
"""

    explanation = get_gemini_response(prompt)

    return {
        "input_product": data_dict,
        "eco_category": eco_category,
        "confidence_percentage": confidence,
        "explanation": explanation
    }