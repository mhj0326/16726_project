from diffusers import StableDiffusionPipeline
import torch

model_id = "Min0326/path-to-save-model_xray"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A x ray images of human lung"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("x_ray1.png")
