import re
import joblib
import numpy as np
from fuzzywuzzy import process, fuzz
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder

# Load all models and encoders
def load_models():
    models = {}
    try:
        models['model'] = joblib.load('models/models/major_recommendation_model.pkl')
        models['mlb_skills'] = joblib.load('models/models/mlb_skills.pkl')
        models['mlb_courses'] = joblib.load('models/models/mlb_courses.pkl')
        models['ohe_work_style'] = joblib.load('models/models/ohe_work_style.pkl')
        models['ohe_passion'] = joblib.load('models/models/ohe_passion.pkl')
        models['le_major'] = joblib.load('models/models/le_major.pkl')
        models['le_faculty'] = joblib.load('models/models/le_faculty.pkl')
        models['le_degree'] = joblib.load('models/models/le_degree.pkl')
        models['le_campus'] = joblib.load('models/models/le_campus.pkl')
        
        # Load master lists
        models['all_skills'] = joblib.load('models/models/master_skills.pkl')
        models['all_courses'] = joblib.load('models/models/master_courses.pkl')
        models['all_passions'] = joblib.load('models/models/master_passions.pkl')
        models['all_work_styles'] = joblib.load('models/models/master_work_styles.pkl')
     
        models['all_skills'].extend([
                'Machine Learning', 'Artificial Intelligence', 'Robotics', 'Programming', 'Database Management',
                'Cloud Computing', 'Biotechnology', 'Environmental Science', 'GIS', 'Engineering Design',
                'Electronics', 'Renewable Energy', 'Lab Safety', 'Photography', 'Video Editing', 'Animation',
                'UX/UI Design', 'Fashion Design', 'Illustration', 'Music Theory', 'Creative Writing',
                'Storyboarding', 'Film Production', 'Problem Solving', 'Statistical Modeling',
                'Research Methodology', 'Project Management', 'Decision Making', 'Risk Assessment',
                'Mathematical Modeling', 'Data Visualization', 'Critical Reading', 'Logical Reasoning',
                'Negotiation', 'Teamwork', 'Coaching', 'Conflict Resolution', 'Presentation Skills',
                'Public Speaking', 'Active Listening', 'Cultural Competence', 'Emotional Intelligence',
                'Customer Relationship Management', 'Spanish', 'French', 'German', 'Mandarin', 'Japanese',
                'Copywriting', 'Editing', 'Technical Writing', 'Translation', 'Speech Writing', 'Blogging',
                'Carpentry', 'Welding', 'Cooking', 'Gardening', 'Mechanical Repairs', 'Automotive Diagnostics',
                'Electrical Wiring', 'Fitness Training', 'Yoga Instruction', 'Meditation Guidance',
                'Meditation Techniques', 'Blockchain', 'Cybersecurity', 'Ethical Hacking', 'Networking Protocols',
                'Mobile App Development', 'Web Development', 'SEO', 'Content Marketing', 'Branding',
                'Public Relations', 'Event Planning', 'Fundraising', 'Sales Strategy', 'Market Research',
                'Negotiation Skills', 'Customer Retention', 'Product Design', 'Industrial Design',
                'Furniture Design', 'Ceramics', 'Pottery', 'Glassblowing', 'Textile Design', 'Fashion Illustration',
                'Stage Design', 'Lighting Design', 'Sound Engineering', 'Acoustics', 'Meteorology', 'Oceanography',
                'Geology', 'Astronomy', 'Astrophysics', 'Genetics', 'Microbiology', 'Pathology',
                'Pharmacology', 'Nutrition', 'Food Science', 'Hospitality Management', 'Tourism Management',
                'Travel Planning', 'Cartography', 'Urban Planning', 'Architecture', 'Interior Design',
                'Landscape Design', 'Sports Coaching', 'Athletic Training', 'Martial Arts', 'First Responder Training',
                'Emergency Management', 'Volunteer Coordination', 'Ethics', 'Philosophy', 'History', 'Sociology',
                'Psychology', 'Anthropology', 'Political Science', 'Law', 'Criminology', 'Forensics', 'Linguistics',
                'Sign Language', 'Calligraphy', 'Typography', 'Meditation', 'Mindfulness', 'Meditation Coaching',
                'Self-defense', 'Survival Skills', 'Animal Care', 'Veterinary Knowledge', 'Aquaculture',
                'Fishing Techniques', 'Hunting Skills', 'Power BI', 'PowerBI', 'Data Analysis', 'Data Analytics',
                'Business Intelligence', 'Excel', 'Tableau'
        ])

        models['all_skills']= list(set(models['all_skills']))
        joblib.dump(models['all_skills'], 'models/models/master_skills.pkl')
     
        print("\n✅ Master skills updated successfully!")
        print("Sample after update:", models['all_skills'][:60])

        #updating the courses list 
        models['all_courses'].extend([
            'Algebra', 'Geometry', 'Trigonometry', 'Calculus', 'Statistics', 'Probability', 
            'Linear Algebra', 'Discrete Mathematics', 'Differential Equations', 'Math', 'Mathematics'

            # Sciences
            'Biology', 'Chemistry', 'Physics', 'Earth Science', 'Environmental Science', 
            'Astronomy', 'Genetics', 'Microbiology', 'Botany', 'Zoology', 'Anatomy', 'Physiology',

            # Languages
            'English', 'Arabic', 'French', 'Spanish', 'German', 'Mandarin', 'Japanese', 'Creative Writing',
            
            # Social Studies / Humanities
            'History', 'World History', 'Geography', 'Economics', 'Political Science', 'Sociology', 'Psychology', 
            'Anthropology', 'Philosophy', 'Ethics', 'Law', 'Civics',

            # Arts
            'Art', 'Drawing', 'Painting', 'Sculpting', 'Music', 'Theater', 'Dance', 'Photography', 'Calligraphy', 
            'Graphic Design', 'Design and Technology',

            # Physical Education
            'Physical Education', 'Health Education', 'Sports', 'Yoga', 'Martial Arts',

            # Technology / Practical Skills
            'Computer Science', 'Information Technology', 'Basic Programming', 'Digital Literacy', 
            'Electronics', 'Engineering Basics', 'Mechanics', 'Home Economics', 'Food and Nutrition', 'Stem Education', 'Robotics',
            
            'Introduction to Computer Science', 'Programming Fundamentals', 'Python Programming', 
            'Java Programming', 'C++ Programming', 'Data Science Fundamentals', 'Machine Learning',
            'Deep Learning', 'Artificial Intelligence', 'Robotics Engineering', 'Database Management Systems',
            'Cloud Computing Basics', 'Cybersecurity Fundamentals', 'Web Development', 'Mobile App Development',
            'Networking Protocols', 'SEO & Content Marketing', 'UX/UI Design', 'Animation Techniques', 'Video Editing',
            
            # Engineering & Science
            'Engineering Design Principles', 'Electronics 101', 'Renewable Energy Technologies', 
            'Environmental Science', 'Biotechnology', 'GIS Mapping', 'Lab Safety', 'Chemistry', 'Physics', 
            'Biology', 'Genetics', 'Microbiology', 'Astrophysics', 'Geology', 'Oceanography', 'Meteorology',
            'Food Science', 'Nutrition', 'Pharmacology', 'Pathology', 'Forensic Science',
            
            # Mathematics
            'Algebra', 'Geometry', 'Trigonometry', 'Calculus', 'Statistics', 'Probability', 'Discrete Mathematics', 
            'Linear Algebra', 'Differential Equations', 'Mathematical Modeling', 'Data Visualization',
            
            # Humanities & Social Sciences
            'History', 'World History', 'Philosophy', 'Ethics', 'Political Science', 'Economics', 'Sociology',
            'Psychology', 'Anthropology', 'Law', 'Criminology', 'Public Speaking', 'Negotiation Skills',
            
            # Arts & Creative
            'Art History', 'Drawing', 'Painting', 'Sculpting', 'Photography Basics', 'Music Theory', 
            'Creative Writing', 'Film Production', 'Stage Design', 'Fashion Design', 'Illustration',
            'Calligraphy', 'Typography', 'Interior Design', 'Landscape Design',
            
            # Practical / Vocational
            'Carpentry', 'Welding', 'Gardening', 'Mechanical Repairs', 'Automotive Diagnostics', 
            'Electrical Wiring', 'Sports Coaching', 'Fitness Training', 'Yoga Instruction', 
            'Martial Arts', 'First Responder Training', 'Emergency Management', 'Volunteer Coordination',
            
            # Business & Management
            'Project Management', 'Decision Making and Strategy', 'Leadership Skills', 
            'Marketing Fundamentals', 'Event Planning', 'Hospitality Management', 'Tourism Management',
            'Sales Strategy', 'Fundraising', 'Customer Relationship Management'])
        models['all_courses']= list(set(models['all_courses']))
        joblib.dump(models['all_courses'], 'models/models/master_courses.pkl')
        print("\n✅ Master courses updated successfully!")
        print("Sample after update:", models['all_courses'][:60])
        models['all_passions'].extend([
                # Arts & Creativity
                'Drawing', 'Painting', 'Sculpting', 'Photography', 'Video Editing', 'Film Making', 'Theater', 
                'Music', 'Singing', 'Playing an Instrument', 'Dance', 'Fashion', 'Design', 'Digital Art', 
                'Graphic Design', 'Animation', 'Storytelling', 'Creative Writing', 'Poetry', 'Calligraphy', 
                'Stage Design', 'Interior Design', 'Landscape Design', 'Ceramics', 'Pottery', 'Crafts', 'DIY Projects',

                # Science & Technology
                'Robotics', 'Astronomy', 'Space Exploration', 'Physics', 'Chemistry', 'Biology', 'Genetics', 
                'Environmental Science', 'Climate Change', 'Mathematics', 'Statistics', 'Computer Science', 
                'Programming', 'Artificial Intelligence', 'Machine Learning', 'Data Science', 'Electronics', 
                'Engineering', 'Renewable Energy', 'Blockchain', 'Cybersecurity', 'Web Development', 
                'Mobile App Development', '3D Printing', 'Tech Gadgets', 'Technology', 'Search', 'Research',

                # Sports & Physical Activities
                'Football', 'Basketball', 'Tennis', 'Swimming', 'Running', 'Cycling', 'Hiking', 'Climbing', 
                'Martial Arts', 'Yoga', 'Meditation', 'Fitness', 'Gym Training', 'Pilates', 'Surfing', 
                'Skiing', 'Skating', 'Horse Riding', 'Rock Climbing', 'Team Sports', 'Outdoor Adventures',

                # Social & Community
                'Volunteering', 'Social Activism', 'Community Service', 'Fundraising', 'Event Planning', 
                'Leadership', 'Mentoring', 'Coaching', 'Public Speaking', 'Debating', 'Negotiation', 
                'Cultural Exchange', 'Networking', 'Human Rights', 'Environmental Activism', 'Politics', 
                'Student Council', 'Teaching', 'Tutoring', 'Youth Programs', 'Animal Rescue', 'Animal Care',

                # Personal Development & Hobbies
                'Reading', 'Book Clubs', 'Writing Blogs', 'Journaling', 'Traveling', 'Learning Languages', 
                'Photography', 'Gardening', 'Cooking', 'Baking', 'Nutrition', 'Mindfulness', 'Meditation Coaching', 
                'Self-defense', 'Survival Skills', 'Fishing', 'Hunting', 'Aquaculture', 'Home Improvement', 
                'Pet Care', 'Fashion Styling', 'Interior Decorating', 'Collecting', 'Chess', 'Puzzles', 
                'Board Games', 'Video Games', 'Virtual Reality',

                # Career & Professional Interests
                'Entrepreneurship', 'Startups', 'Business Management', 'Marketing', 'Sales', 'Customer Service', 
                'Finance', 'Economics', 'Law', 'Criminology', 'Forensics', 'Psychology', 'Counseling', 
                'Human Resources', 'Hospitality Management', 'Tourism', 'Travel Planning', 'Event Management', 
                'Research', 'Laboratory Work', 'Project Management', 'Strategy', 'Data Analysis', 'Innovation'
            ])


        # Remove duplicates
        models['all_passions'] = list(set(models['all_passions']))

        # Save updated passions
        joblib.dump(models['all_passions'], 'models/models/master_passions.pkl')

        print("\n✅ Master passions updated successfully!")
        print("Sample after update:", models['all_passions'][:60])             
        return models
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

models = load_models()

def process_user_text_input(user_input, master_list, threshold=70):
    """
    Enhanced text processing with better spelling mistake handling.
    Uses multiple fuzzy matching strategies to catch spelling errors.
    """
    if not user_input or not isinstance(user_input, str):
        return []
    
    detected_items = set()
    
    # Split by commas, semicolons, or newlines
    tokens = re.split(r'[,\n;]+', user_input.lower())
    
    for token in tokens:
        token = token.strip()
        if not token or len(token) < 2:  # Skip very short tokens
            continue
        
        print(f"🔍 Processing token: '{token}'")
        
        # Strategy 1: Try exact match first (case-insensitive)
        exact_matches = [item for item in master_list if token == item.lower()]
        if exact_matches:
            detected_items.update(exact_matches)
            print(f"   ✅ Exact match found: {exact_matches}")
            continue
        
        # Strategy 2: Try partial match (contains)
        partial_matches = [item for item in master_list if token in item.lower() or item.lower() in token]
        if partial_matches:
            detected_items.update(partial_matches)
            print(f"   ✅ Partial match found: {partial_matches}")
            continue
        
        # Strategy 3: Use multiple fuzzy matching approaches with different thresholds
        
        # Approach A: Token set ratio (good for word order changes)
        matches_a = process.extract(token, master_list, scorer=fuzz.token_set_ratio, limit=5)
        for match, score in matches_a:
            if score >= max(threshold, 80):  # Higher threshold for this scorer
                detected_items.add(match)
                print(f"   ✅ TokenSet match: {match} (score: {score})")
        
        # Approach B: Partial ratio (good for partial matches and spelling errors)
        matches_b = process.extract(token, master_list, scorer=fuzz.partial_ratio, limit=5)
        for match, score in matches_b:
            if score >= threshold:
                detected_items.add(match)
                print(f"   ✅ PartialRatio match: {match} (score: {score})")
        
        # Approach C: WRatio (weighted combination of methods)
        matches_c = process.extract(token, master_list, scorer=fuzz.WRatio, limit=3)
        for match, score in matches_c:
            if score >= threshold:
                detected_items.add(match)
                print(f"   ✅ WRatio match: {match} (score: {score})")
    
    # Final cleanup: Remove any very similar items to avoid duplicates
    final_items = list(detected_items)
    
    print(f"🎯 Final detected items: {final_items}")
    return final_items

def prepare_user_input(user_data):
    """
    Prepare user input for prediction
    """
    if not models:
        return None, "Models not loaded"
    
    # Process RIASEC
    riasec_order = ['R', 'I', 'A', 'S', 'E', 'C']
    X_riasec = np.array([[user_data['riasec'].get(col, 0) for col in riasec_order]])
    
    # Process Skills with NLP - get ALL detected skills
    detected_skills = process_user_text_input(user_data['skills_text'], models['all_skills'], threshold=65)
    print(f"🔍 Final detected skills: {detected_skills}")
    X_skills = models['mlb_skills'].transform([detected_skills])
    
    # Process Courses with NLP - get ALL detected courses
    detected_courses = process_user_text_input(user_data['courses_text'], models['all_courses'], threshold=65)
    print(f"🔍 Final detected courses: {detected_courses}")
    X_courses = models['mlb_courses'].transform([detected_courses])
    
    # Process Work Style
    work_style = user_data['work_style']
    if work_style not in models['all_work_styles']:
        work_style = models['all_work_styles'][0]
    X_work_style = models['ohe_work_style'].transform([[work_style]])
    
    # Process Passion with NLP - get ALL detected passions
    detected_passions = process_user_text_input(user_data['passion_text'], models['all_passions'], threshold=65)
    print(f"🔍 Final detected passions: {detected_passions}")
    passion = detected_passions[0] if detected_passions else models['all_passions'][0]
    X_passion = models['ohe_passion'].transform([[passion]])
    
    # Combine all features
    X_user = np.hstack([X_riasec, X_skills, X_courses, X_work_style, X_passion])
    
    return X_user, {
        'detected_skills': detected_skills,
        'detected_courses': detected_courses,
        'detected_passion': passion
    }

def predict_major(user_data):
    """
    Predict major based on user input
    """
    if not models:
        return {"error": "Models not loaded. Please train the model first."}
    
    # Prepare user input
    X_user, detected_info = prepare_user_input(user_data)
    
    if X_user is None:
        return {"error": "Failed to prepare user input"}
    
    # Make prediction
    try:
        prediction = models['model'].predict(X_user)
        
        # Decode predictions
        result = {
            'major': models['le_major'].inverse_transform([prediction[0][0]])[0],
            'faculty': models['le_faculty'].inverse_transform([prediction[0][1]])[0],
            'degree': models['le_degree'].inverse_transform([prediction[0][2]])[0],
            'campus': models['le_campus'].inverse_transform([prediction[0][3]])[0],
            'detected_info': detected_info,
            'success': True
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}", "success": False}
