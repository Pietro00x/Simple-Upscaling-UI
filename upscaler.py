import gradio as gr
import torch
import numpy as np
from PIL import Image
import cv2
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 
def load_model():
    model_path = 'modelli/RealESRGAN_x4plus.pth'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        device=device
    )
    
    return upsampler

 
def upscale_image(input_img, scale_factor):
    upsampler = load_model()
    
     
    image = Image.fromarray(input_img).convert('RGB')
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
     
    try:
        sr_image, _ = upsampler.enhance(img, outscale=scale_factor)
    except Exception as e:
        print(f"Error during upscaling: {e}")
        raise
    
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(sr_image)

 
def main():
    input_img = gr.Image(label="Upload an image", type="numpy")
    scale_factor = gr.Slider(minimum=2, maximum=4, value=4, step=2, label=" Upscaling Factor")
    output_img = gr.Image(label="Upscaled image", type="pil")
    
    interface = gr.Interface(
        fn=upscale_image,
        inputs=[input_img, scale_factor],
        outputs=output_img,
        title="Simple upscaling UI,
        description=""
    )
    
    interface.launch(server_name="127.0.0.1", server_port=7860)

if __name__ == "__main__":
    main()
