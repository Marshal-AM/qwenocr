from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import torch
import os
from dotenv import load_dotenv

def recognize_text_from_image(image_path, hf_token):
    """
    Recognizes text from an image using Gemma 3n vision model.
    
    Args:
        image_path: Path to the image file
        hf_token: HuggingFace API token for gated model access
    
    Returns:
        Recognized text as string
    """
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    print("Loading image...")
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")
    
    print("Loading Gemma 3n model and processor...")
    print("(This may take a while on first run - downloading ~8GB model)")
    
    model_id = "google/gemma-3n-e4b-it"
    
    # Load model with authentication token
    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=hf_token
    ).eval()
    
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    
    print("Processing image with vision model...")
    
    # Create messages for OCR task
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert at reading and transcribing handwritten text, including medical prescriptions."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Please carefully read and transcribe ALL the text visible in this image. Extract every word, number, and detail you can see, maintaining the structure and layout as much as possible."}
            ]
        }
    ]
    
    # Process inputs
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    input_len = inputs["input_ids"].shape[-1]
    
    print("Generating text from image...")
    
    # Generate response
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=1000,
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
        recognized_text = recognize_text_from_image(image_file, HF_TOKEN)
        
        print("\n" + "="*70)
        print("RECOGNIZED TEXT FROM PRESCRIPTION:")
        print("="*70)
        print(recognized_text)
        print("="*70)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
