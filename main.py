import os
import glob
import requests
import json
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google Custom Search API credentials from .env
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')

def check_env_variables():
    """Check if required environment variables are set"""
    print("\n[ENV CHECK] Checking environment variables...")
    
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
    if not SEARCH_ENGINE_ID:
        raise ValueError("SEARCH_ENGINE_ID not found in .env file")
    
    print(f"[ENV CHECK] ✓ GOOGLE_API_KEY: {GOOGLE_API_KEY[:10]}...{GOOGLE_API_KEY[-4:]}")
    print(f"[ENV CHECK] ✓ SEARCH_ENGINE_ID: {SEARCH_ENGINE_ID}")

def find_prescription_image():
    """Find image file starting with '1' in current directory"""
    print("\n[STEP 0] Searching for prescription image...")
    # Common image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']
    
    for ext in extensions:
        files = glob.glob(f"1{ext}") + glob.glob(f"1.{ext[2:]}")
        if files:
            print(f"[STEP 0] ✓ Found image: {files[0]}")
            return files[0]
    
    raise FileNotFoundError("No image file starting with '1' found in current directory")

def load_model():
    """Load Qwen3-VL model and processor"""
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
    """Parse timing codes like 101, 010, 111 into readable format"""
    timing_map = {
        '1': 'Yes',
        '0': 'No'
    }
    
    if len(code) == 3:
        morning = timing_map.get(code[0], '?')
        afternoon = timing_map.get(code[1], '?')
        night = timing_map.get(code[2], '?')
        
        times = []
        if code[0] == '1':
            times.append('Morning')
        if code[1] == '1':
            times.append('Afternoon')
        if code[2] == '1':
            times.append('Night')
        
        return ', '.join(times) if times else 'Unknown'
    
    return code

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
3. For each medication, identify the timing code (like 101, 010, 111, etc.)

Note: The timing codes mean:
- 101 = Morning and Night
- 010 = Afternoon only
- 111 = Morning, Afternoon, and Night
- First digit = Morning, Second digit = Afternoon, Third digit = Night
- 1 means take medicine, 0 means don't take

Please provide the output in this exact format:
PATIENT NAME: [name]

MEDICATIONS:
1. [Medicine Name] - Timing Code: [code]
2. [Medicine Name] - Timing Code: [code]
...

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
                parts = line.split('-')
                med_name = parts[0].strip()
                # Remove numbering if present
                if med_name and med_name[0].isdigit():
                    med_name = med_name.split('.', 1)[1].strip() if '.' in med_name else med_name
                
                timing_part = parts[1].strip()
                timing_code = timing_part.split(':')[1].strip() if ':' in timing_part else "Unknown"
                
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
            verified_medications.append(med)
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
            'timing_code': med['timing_code'],
            'timing_readable': med['timing_readable']
        }
        
        verified_medications.append(verified_med)
        
        if corrected_name != med_name:
            print(f"[STEP 5] ✓ Corrected: '{med_name}' → '{corrected_name}'")
        else:
            print(f"[STEP 5] ✓ Verified: '{med_name}' is correct")
    
    return verified_medications

def output_final_list(patient_name, verified_medications):
    """STEP 6: Output final verified medication list"""
    print("\n" + "="*60)
    print("[STEP 6] FINAL VERIFIED MEDICATION LIST")
    print("="*60)
    
    print(f"\nPatient Name: {patient_name}\n")
    print("Medications with Intake Times:")
    print("-" * 60)
    
    final_list = []
    for idx, med in enumerate(verified_medications, 1):
        med_info = {
            'number': idx,
            'name': med['corrected_name'],
            'timing_code': med['timing_code'],
            'timing': med['timing_readable']
        }
        final_list.append(med_info)
        
        print(f"{idx}. {med['corrected_name']}")
        print(f"   Timing Code: {med['timing_code']}")
        print(f"   Intake Times: {med['timing_readable']}")
        
        if med['original_name'] != med['corrected_name']:
            print(f"   (Corrected from: {med['original_name']})")
        print()
    
    # Save to file
    output_file = "verified_prescription.json"
    output_data = {
        'patient_name': patient_name,
        'medications': final_list
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"[STEP 6] ✓ Final list saved to: {output_file}")
    
    return final_list

def main():
    try:
        print("\n" + "="*60)
        print("PRESCRIPTION MEDICATION EXTRACTION & VERIFICATION")
        print("="*60)
        
        # Check environment variables
        check_env_variables()
        
        # Find prescription image
        image_path = find_prescription_image()
        
        # Load model once
        model, processor = load_model()
        
        # STEP 1: Extract patient name and medications
        extracted_text = extract_patient_and_medications(image_path, model, processor)
        
        # STEP 2: Parse into list
        patient_name, medications = parse_medications_list(extracted_text)
        
        if not medications:
            print("\n[ERROR] No medications found in prescription!")
            return
        
        # STEP 3 & 4: Google search for each medication
        search_results = search_all_medications(medications)
        
        # STEP 5: Verify medication names with LLM
        verified_medications = verify_medication_names(medications, search_results, model, processor)
        
        # STEP 6: Output final list
        final_list = output_final_list(patient_name, verified_medications)
        
        print("\n" + "="*60)
        print("PROCESS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please ensure your prescription image is named '1.jpg', '1.png', or similar")
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("Please create a .env file with GOOGLE_API_KEY and SEARCH_ENGINE_ID")
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
