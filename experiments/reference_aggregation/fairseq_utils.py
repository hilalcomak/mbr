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
                 tgt_lang: str,
                 model_large: M2M100PreTrainedModel = None,
                 tokenizer_large: M2M100Tokenizer = None,
                 num_subsamples:int = 1):
        self.name = name
        self.model = model
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
        self.tokenizer = tokenizer
        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.pipeline = TranslationPipeline(self.model, self.tokenizer, device=device)
        if num_subsamples > 1:
            assert model_large is not None
            assert tokenizer_large is not None

        self.model_large = model_large 
        self.tokenizer_large = tokenizer_large 
        self.num_subsamples = num_subsamples

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


def load_model(language_pair: str, model: str, subsample_probabilities_model: str, num_subsamples:int) -> TranslationModel:
    name = model.replace('/', '_')
    model_small = M2M100ForConditionalGeneration.from_pretrained(model)
    tokenizer_small = M2M100Tokenizer.from_pretrained(model)
    src_lang, tar_lang = language_pair.split('-')
    assert num_subsamples > 0, 'Requesting less than 1 sub sample makes no sense!'
    if num_subsamples == 1:
      return TranslationModel(name, model_small, tokenizer_small, src_lang, tar_lang)
    assert subsample_probabilities_model is not None, 'Provide a model to get probabilities for subsampling!'
    name = f"{name}-{subsample_probabilities_model}-{num_subsamples}".replace('/', '_')
    model_large = M2M100ForConditionalGeneration.from_pretrained(subsample_probabilities_model)
    tokenizer_large = M2M100Tokenizer.from_pretrained(subsample_probabilities_model)
    return TranslationModel(name, model_small, tokenizer_small, src_lang, tar_lang, model_large, tokenizer_large, num_subsamples)
