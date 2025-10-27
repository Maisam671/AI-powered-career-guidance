from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.utils.ml_utils import predict_major
import logging
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.rag_engine import CareerCompassWeaviate

app = FastAPI(title="Career Compass API")

# Global variable for career system
career_system = None

def initialize_career_system():
    """Initialize the career system on startup"""
    global career_system
    try:
        logger.info("🔄 Initializing Career Compass RAG System...")
        career_system = CareerCompassWeaviate()
        
        # Find the CSV file - CORRECTED PATH
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "final_merged_career_guidance.csv")
        
        logger.info(f"📁 Looking for dataset at: {csv_path}")
        
        if os.path.exists(csv_path):
            logger.info("✅ Dataset found! Initializing RAG system...")
            success = career_system.initialize_system(csv_path)
            if success:
                logger.info("✅ Career Compass RAG System initialized successfully!")
            else:
                logger.warning("⚠️ Career Compass RAG System initialization failed")
                # Still try minimal initialization
                career_system._initialize_minimal()
        else:
            # Initialize minimal system
            logger.warning("📁 No dataset found, initializing minimal system")
            career_system._initialize_minimal()
            
        # Health check
        health = career_system.health_check()
        logger.info(f"🏥 System Health: {health}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize career system: {e}")
        career_system = None

# Initialize on startup
initialize_career_system()

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    work_styles = ["Team-Oriented", "Remote", "On-site", "Office/Data", "Hands-on/Field", "Lab/Research", "Creative/Design", "People-centric/Teaching", "Business", "freelance"]
    return templates.TemplateResponse("index.html", {"request": request, "work_styles": work_styles})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if career_system:
        system_health = career_system.health_check()
        return {
            "message": "Career Compass API is running", 
            "system_health": system_health,
            "status": "healthy"
        }
    else:
        return {
            "message": "Career Compass API is running", 
            "system_health": {"status": "uninitialized"},
            "status": "degraded"
        }

@app.post("/ask")
async def ask_question(data: dict):
    """Chatbot endpoint"""
    try:
        if not career_system:
            return {"answer": "Career system is not available. Please try again later."}
        
        question = data.get("question", "").strip()
        if not question:
            return {"answer": "Please enter a question."}
        
        logger.info(f"❓ Received question: {question}")
        response = career_system.ask_question(question)
        
        logger.info(f"✅ Response confidence: {response.get('confidence', 'Unknown')}")
        return {"answer": response["answer"]}
        
    except Exception as e:
        logger.error(f"❌ Error in ask_question: {e}")
        return {"answer": "Sorry, I'm having trouble processing your question right now. Please try again."}

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
    """ML prediction endpoint"""
    try:
        logger.info("📊 Received prediction request")
        
        riasec = {"R": bool(R), "I": bool(I), "A": bool(A), "S": bool(S), "E": bool(E), "C": bool(C)}
        user_data = {
            "riasec": riasec,
            "skills_text": skills,
            "courses_text": courses,
            "work_style": work_style,
            "passion_text": passion
        }

        result = predict_major(user_data)
        logger.info(f"🎯 Prediction result: {result}")
        
        if "error" in result:
            return JSONResponse({"success": False, "error": result["error"]})
        else:
            result["success"] = True
            return JSONResponse(result)
            
    except Exception as e:
        logger.error(f"❌ Error in predict endpoint: {e}")
        return JSONResponse({"success": False, "error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default to 5000
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
