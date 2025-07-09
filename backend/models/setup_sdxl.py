from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe.save_pretrained("sdxl-base-1.0")
