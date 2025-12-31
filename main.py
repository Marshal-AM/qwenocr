import os
import glob
import requests
import json
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import uvicorn
from pathlib import Path
import shutil

# Load environment variables from .env file
load_dotenv()

# Google Custom Search API credentials from .env
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')

# Create FastAPI app
app = FastAPI(
    title="Prescription Medication Extraction API",
    description="API for extracting and verifying medications from prescription images",
    version="1.0.0"
)

# Global variables to store model (loaded once)
model = None
processor = None

# Directory for uploaded images
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def check_env_variables():
    """Check if required environment variables are set"""
    print("\n[ENV CHECK] Checking environment variables...")
    
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
    if not SEARCH_ENGINE_ID:
        raise ValueError("SEARCH_ENGINE_ID not found in .env file")
    
    print(f"[ENV CHECK] ✓ GOOGLE_API_KEY: {GOOGLE_API_KEY[:10]}...{GOOGLE_API_KEY[-4:]}")
    print(f"[ENV CHECK] ✓ SEARCH_ENGINE_ID: {SEARCH_ENGINE_ID}")

def load_model():
    """Load Qwen3-VL model and processor"""
    global model, processor
    
    if model is None or processor is None:
        print("\n[MODEL LOADING] Loading Qwen3-VL model (this may take a few minutes)...")
        
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct", 
            dtype="auto", 
            device_map="auto"
        )
        
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        
        print("[MODEL LOADING] ✓ Model loaded successfully")
    
    return model, processor

def parse_timing_code(code):
    """Parse timing codes like 101, 010, 111, 1-1-1, 1-0-1 into readable format"""
    # Remove any spaces and convert dashes to empty string for processing
    code_clean = code.replace('-', '').replace(' ', '').strip()
    
    # Handle various formats
    if len(code_clean) == 3 and code_clean.isdigit():
        times = []
        if code_clean[0] == '1':
            times.append('Morning')
        if code_clean[1] == '1':
            times.append('Afternoon')
        if code_clean[2] == '1':
            times.append('Night')
        
        return ', '.join(times) if times else 'Unknown'
    
    return code

def timing_code_to_schedule(timing_code):
    """Convert timing code to schedule format with time and quantity"""
    # Remove any spaces and convert dashes to empty string for processing
    code_clean = timing_code.replace('-', '').replace(' ', '').strip()
    
    schedule = []
    
    # Handle various formats
    if len(code_clean) == 3 and code_clean.isdigit():
        # First digit = Morning, Second digit = Afternoon, Third digit = Evening/Night
        if code_clean[0] == '1':
            schedule.append({
                "time": "MORNING",
                "quantity": "1"
            })
        if code_clean[1] == '1':
            schedule.append({
                "time": "AFTERNOON",
                "quantity": "1"
            })
        if code_clean[2] == '1':
            schedule.append({
                "time": "EVENING",
                "quantity": "1"
            })
    
    # If no schedule created, return empty list
    return schedule if schedule else []

def extract_patient_and_medications(image_path, model, processor):
    """STEP 1: Extract patient name and medications with timing codes"""
    print("\n" + "="*60)
    print("[STEP 1] EXTRACTING PATIENT NAME AND MEDICATIONS")
    print("="*60)
    
    print(f"[STEP 1] Processing image: {image_path}")
    
    # Load image
    image = Image.open(image_path)
    
    # Prepare messages for the model
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text", 
                    "text": """This is a doctor's prescription. Please extract the following information:

1. Patient Name
2. All medications listed in the prescription
3. For each medication, identify the timing code (like 101, 010, 111, 1-1-1, 1-0-1, etc.)

Note: The timing codes mean:
- 1-1-1 or 111 = Morning, Afternoon, and Night
- 1-0-1 or 101 = Morning and Night
- 1-0-0 or 100 = Morning only
- 0-1-0 or 010 = Afternoon only
- 0-0-1 or 001 = Night only
- First digit = Morning, Second digit = Afternoon, Third digit = Night
- 1 means take medicine, 0 means don't take

Please provide the output in this exact format:
PATIENT NAME: [name]

MEDICATIONS:
1. [Medicine Name] - Timing Code: [code]
2. [Medicine Name] - Timing Code: [code]
...

IMPORTANT: Keep the timing code EXACTLY as it appears in the prescription. Do not modify it.
Be very careful to extract the exact medication names and timing codes as they appear."""
                },
            ],
        }
    ]
    
    # Prepare for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    print("[STEP 1] Extracting text from prescription using VLM...")
    
    # Generate output
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=1024,
        do_sample=False,
        repetition_penalty=1.0
    )
    
    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    print("[STEP 1] ✓ Extraction complete")
    print("\n[STEP 1] Raw VLM Output:")
    print("-" * 60)
    print(output_text[0])
    print("-" * 60)
    
    return output_text[0]

def parse_medications_list(extracted_text):
    """STEP 2: Parse extracted text into structured list"""
    print("\n" + "="*60)
    print("[STEP 2] PARSING MEDICATIONS INTO LIST")
    print("="*60)
    
    medications = []
    patient_name = "Unknown"
    
    lines = extracted_text.split('\n')
    
    # Extract patient name
    for line in lines:
        if 'PATIENT NAME:' in line.upper() or 'NAME:' in line.upper():
            patient_name = line.split(':', 1)[1].strip()
            print(f"[STEP 2] Patient Name: {patient_name}")
            break
    
    # Extract medications
    print(f"[STEP 2] Parsing medications...")
    in_medications_section = False
    
    for line in lines:
        line = line.strip()
        
        if 'MEDICATIONS:' in line.upper():
            in_medications_section = True
            continue
        
        if in_medications_section and line:
            # Try to parse medication lines
            # Format: "1. Medicine Name - Timing Code: 101"
            if '-' in line and 'Timing' in line:
                parts = line.split(' - Timing Code:')
                if len(parts) == 2:
                    med_name = parts[0].strip()
                    # Remove numbering if present
                    if med_name and med_name[0].isdigit():
                        med_name = med_name.split('.', 1)[1].strip() if '.' in med_name else med_name
                    
                    timing_code = parts[1].strip()
                    
                    medications.append({
                        'name': med_name,
                        'timing_code': timing_code,
                        'timing_readable': parse_timing_code(timing_code)
                    })
    
    print(f"[STEP 2] ✓ Found {len(medications)} medications")
    print("\n[STEP 2] Medications List:")
    for idx, med in enumerate(medications, 1):
        print(f"  {idx}. {med['name']}")
        print(f"     Timing Code: {med['timing_code']} ({med['timing_readable']})")
    
    return patient_name, medications

def google_search_medication(medication_name):
    """STEP 3: Search medication using Google Custom Search API"""
    print(f"\n[STEP 3] Searching Google for: {medication_name}")
    
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': GOOGLE_API_KEY,
        'cx': SEARCH_ENGINE_ID,
        'q': f"{medication_name} medicine drug"
    }
    
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        results = response.json()
        
        search_results = []
        if 'items' in results:
            for item in results['items'][:5]:  # Top 5 results
                search_results.append({
                    'title': item.get('title', ''),
                    'snippet': item.get('snippet', ''),
                    'link': item.get('link', '')
                })
            print(f"[STEP 3] ✓ Found {len(search_results)} results for {medication_name}")
        else:
            print(f"[STEP 3] ⚠ No results found for {medication_name}")
        
        return search_results
    
    except Exception as e:
        print(f"[STEP 3] ✗ Error searching for {medication_name}: {e}")
        return []

def search_all_medications(medications):
    """STEP 4: Search all medications and return results"""
    print("\n" + "="*60)
    print("[STEP 4] GOOGLE SEARCH FOR ALL MEDICATIONS")
    print("="*60)
    
    all_search_results = {}
    
    for idx, med in enumerate(medications, 1):
        print(f"\n[STEP 4] Processing {idx}/{len(medications)}: {med['name']}")
        search_results = google_search_medication(med['name'])
        all_search_results[med['name']] = search_results
        
        print(f"\n[STEP 4] Search Results for '{med['name']}':")
        print("-" * 60)
        for i, result in enumerate(search_results, 1):
            print(f"  Result {i}:")
            print(f"    Title: {result['title']}")
            print(f"    Snippet: {result['snippet'][:150]}...")
            print()
    
    print(f"[STEP 4] ✓ Completed searches for all {len(medications)} medications")
    return all_search_results

def verify_medication_names(medications, search_results, model, processor):
    """STEP 5: Use LLM to verify and correct medication names"""
    print("\n" + "="*60)
    print("[STEP 5] VERIFYING MEDICATION NAMES WITH LLM")
    print("="*60)
    
    verified_medications = []
    
    for med in medications:
        print(f"\n[STEP 5] Verifying: {med['name']}")
        
        med_name = med['name']
        results = search_results.get(med_name, [])
        
        if not results:
            print(f"[STEP 5] ⚠ No search results available, keeping original name")
            verified_medications.append({
                'original_name': med_name,
                'corrected_name': med_name,
                'timing_code': med['timing_code'],
                'timing_readable': med['timing_readable']
            })
            continue
        
        # Prepare context from search results
        search_context = "\n\n".join([
            f"Result {i+1}:\nTitle: {r['title']}\nSnippet: {r['snippet']}"
            for i, r in enumerate(results[:3])
        ])
        
        # Use LLM to verify the medication name
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""I extracted a medication name from a prescription: "{med_name}"

Here are the Google search results for this medication:

{search_context}

Based on these search results, please:
1. Determine if the extracted medication name is correct or if there's a spelling error
2. If there's an error, provide the correct medication name
3. If the name is correct, confirm it

Please respond in this exact format:
ORIGINAL: [original name]
CORRECT: [corrected name]
CONFIDENCE: [High/Medium/Low]
REASONING: [brief explanation]

Only provide the corrected name without any additional information like dosage or form."""
                    }
                ]
            }
        ]
        
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        print(f"[STEP 5] Analyzing with LLM...")
        
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.0
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        llm_output = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"[STEP 5] LLM Analysis:")
        print(f"{llm_output}")
        
        # Parse LLM output to get corrected name
        corrected_name = med_name  # Default to original
        for line in llm_output.split('\n'):
            if 'CORRECT:' in line.upper():
                corrected_name = line.split(':', 1)[1].strip()
                break
        
        verified_med = {
            'original_name': med_name,
            'corrected_name': corrected_name,
            'timing_code': med['timing_code'],  # Keep original timing code AS IS
            'timing_readable': med['timing_readable']  # Keep original timing readable AS IS
        }
        
        verified_medications.append(verified_med)
        
        if corrected_name != med_name:
            print(f"[STEP 5] ✓ Corrected: '{med_name}' → '{corrected_name}'")
        else:
            print(f"[STEP 5] ✓ Verified: '{med_name}' is correct")
    
    return verified_medications

def build_final_list(patient_name, verified_medications):
    """STEP 6: Build final verified medication list"""
    print("\n" + "="*60)
    print("[STEP 6] FINAL VERIFIED MEDICATION LIST")
    print("="*60)
    
    print(f"\nPatient Name: {patient_name}\n")
    print("Medications with Intake Times:")
    print("-" * 60)
    
    final_list = []
    for idx, med in enumerate(verified_medications, 1):
        # Convert timing code to schedule format
        schedule = timing_code_to_schedule(med['timing_code'])
        
        med_info = {
            'medication': med['corrected_name'],
            'schedule': schedule
        }
        final_list.append(med_info)
        
        print(f"{idx}. {med['corrected_name']}")
        print(f"   Timing Code: {med['timing_code']}")
        print(f"   Intake Times: {med['timing_readable']}")
        print(f"   Schedule: {schedule}")
        
        if med['original_name'] != med['corrected_name']:
            print(f"   (Corrected from: {med['original_name']})")
        print()
    
    print(f"[STEP 6] ✓ Final list created")
    
    return final_list

def process_prescription_pipeline(image_path: str) -> Dict[str, Any]:
    """Main processing pipeline"""
    try:
        print("\n" + "="*60)
        print("PRESCRIPTION MEDICATION EXTRACTION & VERIFICATION")
        print("="*60)
        
        # Check environment variables
        check_env_variables()
        
        # Load model
        model, processor = load_model()
        
        # STEP 1: Extract patient name and medications
        extracted_text = extract_patient_and_medications(image_path, model, processor)
        
        # STEP 2: Parse into list
        patient_name, medications = parse_medications_list(extracted_text)
        
        if not medications:
            raise ValueError("No medications found in prescription!")
        
        # STEP 3 & 4: Google search for each medication
        search_results = search_all_medications(medications)
        
        # STEP 5: Verify medication names with LLM
        verified_medications = verify_medication_names(medications, search_results, model, processor)
        
        # STEP 6: Build final list
        final_list = build_final_list(patient_name, verified_medications)
        
        print("\n" + "="*60)
        print("PROCESS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return {
            "status": "success",
            "patient_name": patient_name,
            "medications": final_list,
            "total_medications": len(final_list)
        }
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise

# FastAPI Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Prescription Medication Extraction API",
        "version": "1.0.0",
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/extract": "Extract medications from prescription (POST)",
            "/docs": "API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and processor is not None,
        "google_api_configured": GOOGLE_API_KEY is not None and SEARCH_ENGINE_ID is not None
    }

@app.post("/extract")
async def extract_prescription(file: UploadFile = File(...)):
    """
    Extract medications from a prescription image
    
    Args:
        file: Prescription image file (jpg, png, etc.)
    
    Returns:
        JSON with patient name and verified medication list
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file
        file_extension = Path(file.filename).suffix
        file_path = UPLOAD_DIR / f"prescription_{file.filename}"
        
        print(f"\n[API] Saving uploaded file: {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the prescription
        print(f"[API] Processing prescription...")
        result = process_prescription_pipeline(str(file_path))
        
        # Clean up uploaded file
        print(f"[API] Cleaning up uploaded file...")
        file_path.unlink()
        
        return JSONResponse(content=result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("\n" + "="*60)
    print("STARTING PRESCRIPTION EXTRACTION API")
    print("="*60)
    try:
        check_env_variables()
        load_model()
        print("\n[API] ✓ API ready to accept requests")
    except Exception as e:
        print(f"\n[API] ✗ Error during startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n[API] Shutting down...")
    # Clean up any remaining files in uploads directory
    for file in UPLOAD_DIR.glob("*"):
        file.unlink()
    print("[API] ✓ Cleanup complete")

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "main:app",  # Change "main" to your script filename if different
        host="0.0.0.0",
        port=8000,
        reload=False  # Set to True for development
    )
