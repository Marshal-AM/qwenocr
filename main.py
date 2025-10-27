import os
import glob
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

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
    """Extract text from prescription image using Qwen3-VL"""
    
    print("Loading Qwen3-VL model (this may take a few minutes)...")
    
    # Load model and processor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct", 
        dtype="auto", 
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    
    print(f"Processing image: {image_path}")
    
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
                    "text": "This is a doctor's prescription. Please extract all the text from this prescription image, including patient details, doctor's name, medications, dosages, instructions, and any other relevant information. Provide the output in a clear, structured format."
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
    
    print("Extracting text from prescription...")
    
    # Generate output with optimized parameters for OCR
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=2048,
        do_sample=False,  # Greedy decoding for more accurate OCR
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
    
    return output_text[0]

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

if __name__ == "__main__":
    main()
