import os
import glob
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image

# Get HuggingFace token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HuggingFace token not found! Please set the HF_TOKEN environment variable.\n"
        "You can set it by running: export HF_TOKEN='your_token_here'"
    )

def find_prescription_image():
    """Find image file starting with '1' in current directory"""
    # Common image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']
    
    for ext in extensions:
        files = glob.glob(f"1{ext}") + glob.glob(f"1.{ext[2:]}")
        if files:
            return files[0]
    
    raise FileNotFoundError("No image file starting with '1' found in current directory")

def extract_prescription_text(image_path):
    """Extract text from prescription image using Llama 3.2 Vision"""
    
    print("Loading Llama 3.2 Vision 11B model (this may take a few minutes)...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
    # Load model and processor with HF token
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN
    )
    
    processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
    
    print(f"Model loaded on device: {model.device}")
    print(f"Processing image: {image_path}")
    
    # Load image
    image = Image.open(image_path)
    
    # Prepare messages for the model
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image"},
                {
                    "type": "text", 
                    "text": "This is a doctor's prescription. Please extract all the text from this prescription image, including patient details, doctor's name, medications, dosages, instructions, and any other relevant information. Provide the output in a clear, structured format."
                }
            ]
        }
    ]
    
    # Prepare input
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    
    print("Extracting text from prescription...")
    
    # Generate output
    output = model.generate(**inputs, max_new_tokens=2048)
    
    # Decode output
    extracted_text = processor.decode(output[0], skip_special_tokens=True)
    
    # Remove the prompt from the output (keep only the generated response)
    # The output includes the input prompt, so we extract only the assistant's response
    if "assistant" in extracted_text:
        extracted_text = extracted_text.split("assistant")[-1].strip()
    
    return extracted_text

def main():
    try:
        # Find the prescription image
        image_path = find_prescription_image()
        
        # Extract text from prescription
        extracted_text = extract_prescription_text(image_path)
        
        # Display results
        print("\n" + "="*60)
        print("EXTRACTED PRESCRIPTION TEXT")
        print("="*60)
        print(extracted_text)
        print("="*60)
        
        # Save to file
        output_file = "prescription_extracted.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        
        print(f"\nExtracted text saved to: {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure your prescription image is named '1.jpg', '1.png', or similar")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
