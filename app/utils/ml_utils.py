import re
import joblib
import numpy as np
from fuzzywuzzy import process, fuzz
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder

logger = logging.getLogger(__name__)

# Global models variable with lazy loading
_models = None

def get_models():
    global _models
    if _models is None:
        logger.info("🔄 Loading ML models...")
        _models = load_models()
    return _models

def load_models():
    """Load models with minimal memory footprint"""
    try:
        models = {}
        base_path = "models/models/"
        
        # Only load essential models
        essential_models = [
            'major_recommendation_model.pkl',
            'mlb_skills.pkl', 
            'mlb_courses.pkl',
            'ohe_work_style.pkl',
            'ohe_passion.pkl',
            'le_major.pkl',
            'le_faculty.pkl', 
            'le_degree.pkl',
            'le_campus.pkl',
            'master_skills.pkl',
            'master_courses.pkl',
            'master_passions.pkl',
            'master_work_styles.pkl'
        ]
        
        for model_file in essential_models:
            key = model_file.replace('.pkl', '').replace('master_', '')
            if 'master_' in model_file:
                key = f"all_{key}"
            models[key] = joblib.load(f'{base_path}{model_file}')
            logger.info(f"✅ Loaded {model_file}")
        
        logger.info("🎯 All ML models loaded successfully")
        return models
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        return None


def process_user_text_input(user_input, master_list, input_type="skills"):
    """
    Enhanced text processing with better handling of comma-separated values
    """
    if not user_input or not isinstance(user_input, str):
        return []
    
    detected_items = set()
    
    print(f"🔍 Processing {input_type} input: '{user_input}'")
    
    # Clean and normalize the input
    user_input = user_input.strip()
    
    # Handle comma-separated values more intelligently
    tokens = []
    if ',' in user_input:
        # Split by commas but be careful with spaces
        parts = [part.strip() for part in user_input.split(',')]
        tokens.extend(parts)
    else:
        # Also split by spaces for single entries
        tokens = [user_input]
    
    # Remove empty tokens
    tokens = [token for token in tokens if token and len(token) > 1]
    
    print(f"   Tokens extracted: {tokens}")
    
    for token in tokens:
        token = token.strip()
        if not token:
            continue
            
        print(f"   🔎 Matching token: '{token}'")
        
        # Strategy 1: Exact match (case-insensitive)
        exact_matches = [item for item in master_list if token.lower() == item.lower()]
        if exact_matches:
            detected_items.update(exact_matches)
            print(f"      ✅ Exact match: {exact_matches}")
            continue
        
        # Strategy 2: Partial match
        partial_matches = [item for item in master_list if token.lower() in item.lower() or item.lower() in token.lower()]
        if partial_matches:
            # Take the best partial match (longest or most specific)
            best_match = max(partial_matches, key=len)
            detected_items.add(best_match)
            print(f"      ✅ Partial match: {best_match}")
            continue
        
        # Strategy 3: Fuzzy matching with multiple approaches
        # Try WRatio first (balanced approach)
        matches = process.extract(token, master_list, scorer=fuzz.WRatio, limit=10)
        for match, score in matches:
            if score >= 75:  # Good balance for spelling errors
                detected_items.add(match)
                print(f"      ✅ Fuzzy match: {match} (score: {score})")
                break  # Take the best fuzzy match for this token
    
    final_items = list(detected_items)
    print(f"🎯 Final {input_type}: {final_items}")
    return final_items

def prepare_user_input(user_data):
    """
    Prepare user input for prediction with extensive debugging
    """
    if not models:
        print("❌ Models not loaded!")
        return None, "Models not loaded"
    
    print(f"📥 Received user data: {user_data}")
    
    # Process RIASEC
    riasec_order = ['R', 'I', 'A', 'S', 'E', 'C']
    riasec_values = [1 if user_data['riasec'].get(col, False) else 0 for col in riasec_order]
    X_riasec = np.array([riasec_values])
    print(f"🎭 RIASEC values: {dict(zip(riasec_order, riasec_values))}")
    
    # Process Skills
    skills_text = user_data.get('skills_text', '')
    detected_skills = process_user_text_input(skills_text, models['all_skills'], "skills")
    
    # If no skills detected but we have text, try to extract individual words
    if not detected_skills and skills_text:
        print("🔄 Trying alternative skill extraction...")
        # Extract individual words and try to match them
        words = re.findall(r'\b\w+\b', skills_text.lower())
        for word in words:
            if len(word) > 3:  # Only consider words longer than 3 characters
                matches = process.extract(word, models['all_skills'], scorer=fuzz.partial_ratio, limit=3)
                for match, score in matches:
                    if score >= 80:
                        detected_skills.append(match)
                        break
        detected_skills = list(set(detected_skills))
        print(f"🔄 Alternative skills detected: {detected_skills}")
    
    X_skills = models['mlb_skills'].transform([detected_skills])
    print(f"🔧 Skills features shape: {X_skills.shape}")
    
    # Process Courses
    courses_text = user_data.get('courses_text', '')
    detected_courses = process_user_text_input(courses_text, models['all_courses'], "courses")
    X_courses = models['mlb_courses'].transform([detected_courses])
    print(f"🔧 Courses features shape: {X_courses.shape}")
    
    # Process Work Style
    work_style = user_data.get('work_style', '')
    print(f"💼 Work style: {work_style}")
    if work_style not in models['all_work_styles']:
        work_style = models['all_work_styles'][0] if models['all_work_styles'] else 'Office/Data'
        print(f"🔄 Using default work style: {work_style}")
    X_work_style = models['ohe_work_style'].transform([[work_style]])
    print(f"🔧 Work style features shape: {X_work_style.shape}")
    
    # Process Passion
    passion_text = user_data.get('passion_text', '')
    detected_passions = process_user_text_input(passion_text, models['all_passions'], "passions")
    passion = detected_passions[0] if detected_passions else (models['all_passions'][0] if models['all_passions'] else 'Technology')
    print(f"❤️ Passion: {passion}")
    X_passion = models['ohe_passion'].transform([[passion]])
    print(f"🔧 Passion features shape: {X_passion.shape}")
    
    # Combine all features
    X_user = np.hstack([X_riasec, X_skills, X_courses, X_work_style, X_passion])
    print(f"🎯 Final feature vector shape: {X_user.shape}")
    
    detected_info = {
        'detected_skills': detected_skills,
        'detected_courses': detected_courses,
        'detected_passion': passion
    }
    
    print(f"📊 Detected info: {detected_info}")
    
    return X_user, detected_info

def predict_major(user_data):
    """
    Predict major based on user input
    """
    models = get_models()
    if not models:
        return {"error": "Models not loaded. Please train the model first."}
    
    print("🎯 Starting prediction...")
    
    # Prepare user input
    X_user, detected_info = prepare_user_input(user_data)
    
    if X_user is None:
        return {"error": "Failed to prepare user input"}
    
    # Make prediction
    try:
        prediction = models['model'].predict(X_user)
        print(f"🤖 Model prediction: {prediction}")
        
        # Decode predictions
        result = {
            'major': models['le_major'].inverse_transform([prediction[0][0]])[0],
            'faculty': models['le_faculty'].inverse_transform([prediction[0][1]])[0],
            'degree': models['le_degree'].inverse_transform([prediction[0][2]])[0],
            'campus': models['le_campus'].inverse_transform([prediction[0][3]])[0],
            'detected_info': detected_info,
            'success': True
        }
        
        print(f"✅ Prediction result: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error": error_msg, "success": False}