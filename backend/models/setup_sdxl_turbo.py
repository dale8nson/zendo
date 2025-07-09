from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
pipe.save_pretrained("sdxl-base-1.0-turbo")
