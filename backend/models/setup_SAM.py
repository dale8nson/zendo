from transformers import SamModel, SamProcessor

model = SamModel.from_pretrained("facebook/sam-vit-base")
model.save_pretrained("sam_model")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
processor.save_pretrained("sam_processor")
