from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import os
from dotenv import load_dotenv

def recognize_prescription_text(image_path, hf_token):
    """
    Recognizes text from a prescription image using MedGemma 4B model.
    MedGemma is specifically trained on medical images and text.
    
    Args:
        image_path: Path to the prescription image file
        hf_token: HuggingFace API token for model access
    
    Returns:
        Recognized text as string
    """
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    print("Loading prescription image...")
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")
    
    print("\nLoading MedGemma 4B model and processor...")
    print("(This may take a while on first run - downloading ~4GB model)")
    
    model_id = "google/medgemma-4b-it"
    
    # Load model with authentication token
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token
    )
    
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    
    print("Processing prescription with medical vision model...")
    
    # Create messages optimized for prescription OCR
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert medical document reader specializing in reading prescriptions and medical handwriting."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please carefully read and transcribe ALL text visible in this prescription image. Extract every detail including: patient information, medication names, dosages, instructions, doctor's name, date, and any other text you can see. Preserve the structure and layout as much as possible."},
                {"type": "image", "image": image}
            ]
        }
    ]
    
    # Process inputs
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    
    input_len = inputs["input_ids"].shape[-1]
    
    print("Generating transcription from prescription...")
    
    # Generate response with higher token limit for detailed transcriptions
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=1500,
            do_sample=False,
            temperature=0.1
        )
        generation = generation[0][input_len:]
    
    # Decode the generated text
    decoded = processor.decode(generation, skip_special_tokens=True)
    
    return decoded

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Get HuggingFace token from environment
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN not found in environment variables. "
            "Please create a .env file with: HF_TOKEN=your_token_here"
        )
    
    # Image file name in project root directory
    image_file = "1"
    
    # Try common image extensions if no extension provided
    if not any(image_file.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.PNG']:
            if os.path.exists(image_file + ext):
                image_file = image_file + ext
                break
    
    try:
        recognized_text = recognize_prescription_text(image_file, HF_TOKEN)
        
        print("\n" + "="*70)
        print("PRESCRIPTION TRANSCRIPTION (MedGemma 4B):")
        print("="*70)
        print(recognized_text)
        print("="*70)
        print("\nNote: This is a medical AI model output. Always verify")
        print("transcriptions with qualified healthcare professionals.")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
