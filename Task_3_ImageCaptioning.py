import warnings
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

warnings.filterwarnings("ignore")

class ImageCaptioner:
    def __init__(self):
        print("Loading models... This may take a minute first time.")
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        
        # Set device to CPU (or GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print("Model loaded successfully!")

    def generate_caption(self, image_path):
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert(strict=True, mode="RGB")

            pixel_values = self.feature_extractor(images=[image], return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            output_ids = self.model.generate(pixel_values, max_length=16, num_beams=4)

            preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            return preds[0].strip()
            
        except Exception as e:
            return f"Error processing image: {str(e)}"

def main():
    captioner = ImageCaptioner()
    
    print("\n" + "="*50)
    print("AI Image Captioning System")
    print("="*50)
    print("Instructions: Type the name of an image file (e.g., 'cat.jpg') to caption it.")
    print("Type 'exit' to quit.")
    
    while True:
        image_path = input("\nEnter image path: ").strip()
        
        if image_path.lower() == 'exit':
            break

        image_path = image_path.replace('"', '').replace("'", "")
        
        caption = captioner.generate_caption(image_path)
        print(f"\nAI sees: '{caption}'")

if __name__ == "__main__":
    main()