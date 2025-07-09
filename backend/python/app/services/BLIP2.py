from transformers import Blip2VisionConfig, Blip2QFormerConfig, OPTConfig, Blip2Config, Blip2ForConditionalGeneration, Blip2Processor



configuration = None
class Blip2Model:
    def __init__(self):

        self.configuration = Blip2Config()
        self.model = Blip2ForConditionalGeneration(configuration)
        self.configuration = self.model.config
        self.vision_config = Blip2VisionConfig()
        self.qformer_config = Blip2QFormerConfig()
        self.text_config = OPTConfig()
        self.config = Blip2Config.from_vision_qformer_text_configs(self.vision_config, self.qformer_config,self. text_config)
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
