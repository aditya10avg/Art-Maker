import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

# Load the pre-trained model from Hugging Face
model_id = "runwayml/stable-diffusion-v1-5"  # You can replace it with other available models

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

# Load and preprocess the input image
def preprocess_image(image_path, target_size=(512, 512)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    return image

input_image = preprocess_image("input.jpg")  # Path to your input image

# Set the prompt for the art style
prompt = "an artistic painting, in the style of Van Gogh"  # You can change the style or prompt here

# Generate the art form from the input image
with torch.autocast(device):
    output = pipe(prompt=prompt, init_image=input_image, strength=0.75, guidance_scale=7.5)

# Save the generated image
output.images[0].save("output_art.jpg")
