# main.py (in ROOT directory)
import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=port, 
        workers=1,  # Use only 1 worker to save memory
        log_level="info"
    )