from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from datetime import datetime

app = FastAPI(title="AetherEU - Sovereign AI Demo (No Torch)")

class Query(BaseModel):
    text: str
    target_language: str = "en"
    task: str = "general"

# Expanded Mock Responses for EU Bureaucratic Tasks
MOCK_RESPONSES = {
    "erasmus": """✅ **Erasmus+ for Polish Students**

1. Contact your university's International Office
2. Register on the Erasmus+ platform
3. Prepare documents:
   - Learning Agreement
   - Transcript of Records
   - Motivation Letter (in English or host language)
   - Europass CV
4. Application deadline: usually February–March
5. Funding: 300–600€/month + travel grant

Would you like a sample **Motivation Letter in Polish**?""",

    "fund": """✅ **EU Funds & Grants**

You are likely eligible.
Next steps:
- Check current open calls on funding.europa.eu
- Contact your National Agency
- Submit via your university/organisation (they manage the funds)""",

    "welfare": """✅ **Welfare Benefits (e.g. 800+ / Rodzina 800+ in Poland)**

1. Apply online via gov.pl or PUE ZUS
2. Required: PESEL, bank account, child's birth certificate
3. Payment: automatically transferred monthly
4. Income test usually not required for basic 800+""",

    "germany": """✅ **German Benefits (Bürgergeld / Kindergeld)**

- Kindergeld: €250 per child/month
- Apply via Familienkasse
- Cross-border: Possible while working in Germany even if living in Poland (EU rules)""",

    "tax": """✅ **Tax Declaration (EU Cross-border)**

- Use your country's tax portal (e.g. Twój e-PIT in Poland)
- Report foreign income via double taxation treaty
- Deadline: usually April 30th
- You may get tax refund from both countries""",

    "visa": """✅ **EU Visa / Residence Permit**

- For non-EU family members: Apply for EU Family Residence Card
- Processing time: 3–6 months
- Required: marriage/birth certificate + proof of sufficient resources""",

    "pension": """✅ **EU Pension / Retirement**

You can combine pension periods from different EU countries.
- Use the EU Pension Tracker or contact ZUS (Poland) / Deutsche Rentenversicherung
- Request S1 form for healthcare coordination"""
}

@app.post("/v1/eu-ai/query")
async def query_ai(query: Query):
    text_lower = query.text.lower()
    task_lower = query.task.lower()
    
    # Smart routing to best mock response
    if any(word in text_lower for word in ["erasmus", "exchange", "study abroad"]):
        answer = MOCK_RESPONSES["erasmus"]
    elif any(word in text_lower + task_lower for word in ["fund", "grant", "funding"]):
        answer = MOCK_RESPONSES["fund"]
    elif any(word in text_lower for word in ["welfare", "benefit", "800+", "zasiłek", "rodzina"]):
        answer = MOCK_RESPONSES["welfare"]
    elif any(word in text_lower for word in ["germany", "niemcy", "kindergeld", "bürgergeld"]):
        answer = MOCK_RESPONSES["germany"]
    elif any(word in text_lower for word in ["tax", "podatek", "vat", "declaration"]):
        answer = MOCK_RESPONSES["tax"]
    elif any(word in text_lower for word in ["visa", "residence", "permit", "schengen"]):
        answer = MOCK_RESPONSES["visa"]
    elif any(word in text_lower for word in ["pension", "emerytura", "retirement"]):
        answer = MOCK_RESPONSES["pension"]
    else:
        answer = f"""✅ **AetherEU Answer** (in {query.target_language})

Query: "{query.text}"

**Full sovereign version would provide:**
• Step-by-step official process
• Required documents with direct links
• Current deadlines
• Pre-filled forms
• Eligibility check based on your situation"""

    return {
        "success": True,
        "answer": answer,
        "language": query.target_language,
        "timestamp": datetime.now().isoformat(),
        "model": "AetherEU-NoTorch-Demo-v2.1",
        "confidence": 0.89
    }

@app.get("/v1/eu-ai/languages")
async def get_languages():
    return {
        "supported_languages": 24,
        "status": "ready",
        "message": "All official EU languages supported in full version"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "AetherEU Lightweight Demo v2.1"}

if __name__ == "__main__":
    print("🚀 Starting AetherEU Sovereign AI (No Torch - Super Light v2.1)")
    print("   Server running at http://localhost:8000")
    print("   More mock responses added (Erasmus, Welfare, Tax, Visa, etc.)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
