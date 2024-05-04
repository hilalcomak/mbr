import urllib
from pathlib import Path
from typing import Union, List

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, M2M100PreTrainedModel, TranslationPipeline


class TranslationModel:
    def __init__(self,
                 name: str,
                 model: M2M100PreTrainedModel,
                 tokenizer: M2M100Tokenizer,
                 src_lang: str,
                 tgt_lang: str):
        self.name = name
        self.model = model
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
        self.tokenizer = tokenizer
        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.pipeline = TranslationPipeline(self.model, self.tokenizer, device=device)

    def translate(self, sentences: List[str], beam: int = 5, **kwargs) -> List[str]:
        self.pipeline.batch_size = len(sentences)
        outputs = self.pipeline(
            sentences,
            src_lang=self.tokenizer.src_lang,
            tgt_lang=self.tokenizer.tgt_lang,
            num_beams=beam,
            **kwargs,
        )
        translations = [output['translation_text'] for output in outputs]
        return translations

    def sample(self, sentences: List[str], seed=None, **kwargs) -> List[str]:
        outputs = self.pipeline(
            sentences,
            src_lang=self.tokenizer.src_lang,
            tgt_lang=self.tokenizer.tgt_lang,
            do_sample=1,
            **kwargs,
        )
        translations = [output['translation_text'] for output in outputs]
        return translations

    def __str__(self):
        return self.pipeline()


def load_model(language_pair: str) -> TranslationModel:
    name = 'facebook/m2m100_418M'
    model = M2M100ForConditionalGeneration.from_pretrained(name)
    tokenizer = M2M100Tokenizer.from_pretrained(name)
    src_lang, tar_lang = language_pair.split('-')
    return TranslationModel(name, model, tokenizer, src_lang, tar_lang)