import gradio as gr
import numpy as np
from PIL import Image
import torch
import torch.optim as optim

from model_architecture.architecture import Generator
from dataset.map_dataset import transform_only_input, both_transform

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

# Initialize generator model
gen = Generator(input_channels=3).to("cpu")
opt_gen = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))

LEARNING_RATE = 2e-4
CHECKPOINT_GEN = "./Pix2PixModel/gen1.pth.tar"

# Load checkpoint
load_checkpoint(CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE)

# Global variable to store the full image
full_image_data = None

def update_satellite_view(image):
    """Store the full image but show only satellite portion"""
    global full_image_data
    
    if image is None:
        full_image_data = None
        return None
    
    # Store the complete image for later use
    full_image_data = image
    
    # Extract and return only the satellite portion
    img_array = np.array(image)
    satellite_portion = img_array[:, :600, :]
    return Image.fromarray(satellite_portion)

def generate_map_view():
    """Generate map view using the stored full image"""
    global full_image_data
    
    if full_image_data is None:
        return None
    
    image = np.array(full_image_data)
    
    # Split into satellite and map parts
    input_image = image[:, :600, :]
    target_image = image[:, 600:, :]

    augmentations = both_transform(image=input_image, image0=target_image)
    input_image, _ = augmentations["image"], augmentations["image0"]

    input_image = transform_only_input(image=input_image)["image"]
    input_image = input_image.unsqueeze(0)

    gen.eval()
    with torch.no_grad():
        y_fake = gen(input_image)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization
        map_view = y_fake

    map_view = map_view.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Convert to uint8 for PIL
    map_view = (map_view * 255).astype(np.uint8)
    
    # Convert to PIL Image
    map_view = Image.fromarray(map_view)

    return map_view

custom_css = """
#generate-button {
    background-color: #4CAF50 !important; /* Green background */
    color: white !important;             /* White text */
    border: none !important;
    padding: 10px 24px !important;
    text-align: center !important;
    text-decoration: none !important;
    display: inline-block !important;
    font-size: 16px !important;
    margin: 4px 2px !important;
    cursor: pointer !important;
    border-radius: 8px !important;
}
"""

with gr.Blocks(title="Satellite to Map View Generator", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("<h2 style='text-align: center;'>Generate Map View from Satellite Image</h2>")
    
    with gr.Column():
        # Display only the satellite portion
        input_image = gr.Image(label="Satellite View", type="pil")
        submit_button = gr.Button("Generate Map View", elem_id="generate-button")
        output_image = gr.Image(label="Generated Map View")
    
    # Process the upload to show only satellite portion
    input_image.upload(fn=update_satellite_view, inputs=input_image, outputs=input_image)
    
    # Generate map when button is clicked
    submit_button.click(fn=generate_map_view, inputs=None, outputs=output_image)

demo.launch()