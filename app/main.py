import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import logging
import uvicorn

# Load environment variables FIRST
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files with error handling
try:
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
except:
    logger.warning("Static files directory not found")

try:
    templates = Jinja2Templates(directory="app/templates")
except:
    logger.warning("Templates directory not found")

# Initialize RAG system LAZILY (don't load on startup)
career_system = None

def get_career_system():
    global career_system
    if career_system is None:
        try:
            from app.rag_engine import CareerCompassWeaviate
            career_system = CareerCompassWeaviate()
            # Don't initialize the full system on import to save memory
        except Exception as e:
            logger.error(f"Failed to initialize career system: {e}")
    return career_system

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Career Compass is running"}

@app.get("/")
async def root():
    return {"message": "Career Compass API is running"}

# Home page - ML Recommendation
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    work_styles = ["Team-Oriented","Remote", "On-site","Office/Data", "Hands-on/Field","Lab/Research","Creative/Design", "People-centric/Teaching", "Business", "freelance"]
    return templates.TemplateResponse("index.html", {"request": request, "work_styles": work_styles})

# Chatbot API
@app.post("/ask")
async def ask_question(data: dict):
    try:
        q = data.get("question")
        logger.info(f"Received question: {q}")
        
        system = get_career_system()
        if not system:
            return {"answer": "Career guidance system is starting up. Please try again in a moment."}
            
        response = system.ask_question(q)
        return {"answer": response["answer"]}
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        return {"answer": "Sorry, I'm having trouble processing your question right now."}

# ML Prediction Endpoint
@app.post("/predict")
async def predict(
    request: Request,
    R: str = Form(None),
    I: str = Form(None),
    A: str = Form(None),
    S: str = Form(None),
    E: str = Form(None),
    C: str = Form(None),
    skills: str = Form(""),
    courses: str = Form(""),
    work_style: str = Form(""),
    passion: str = Form("")
):
    try:
        logger.info(f"📥 Form data received:")
        logger.info(f"  RIASEC: R={R}, I={I}, A={A}, S={S}, E={E}, C={C}")
        
        # Convert checkbox values to boolean
        riasec = {
            "R": bool(R), 
            "I": bool(I), 
            "A": bool(A), 
            "S": bool(S), 
            "E": bool(E), 
            "C": bool(C)
        }
        
        user_data = {
            "riasec": riasec,
            "skills_text": skills,
            "courses_text": courses,
            "work_style": work_style,
            "passion_text": passion
        }

        from app.utils.ml_utils import predict_major
        result = predict_major(user_data)
        logger.info(f"📤 Prediction completed")
        
        if "error" in result:
            return JSONResponse({"success": False, "error": result["error"]})
        else:
            result["success"] = True
            return JSONResponse(result)
            
    except Exception as e:
        logger.error(f"❌ Error in predict endpoint: {e}")
        return JSONResponse({"success": False, "error": str(e)})

# This is crucial for Render
if __name__ in "__main__":
    app.run(host="0.0.0.0", port=5000,debug=False)
