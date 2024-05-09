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
        self.model_large = model_large
        self.tokenizer_large = tokenizer_large 
        self.num_subsamples = num_subsamples
        self.device = device
        if num_subsamples > 1:
            assert model_large is not None
            assert tokenizer_large is not None
            self.model_large = model_large.to(device)
            self.tokenizer_large = tokenizer_large

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

    @torch.no_grad()
    def sample(self, num_samples:int , source_sentence: str, seed=None, **kwargs) -> List[str]:
        sentences = (num_samples * self.num_subsamples) * [source_sentence]
        outputs = self.pipeline(
            sentences,
            src_lang=self.tokenizer.src_lang,
            tgt_lang=self.tokenizer.tgt_lang,
            do_sample=1,
            **kwargs,
        )
        translations = [output['translation_text'] for output in outputs]
        if self.num_subsamples > 1:
            losses = []
            MAX_PARALLEL_LARGE_MODEL = 10
            for i in range(len(sentences)//MAX_PARALLEL_LARGE_MODEL):
                model_inputs = self.tokenizer_large(sentences[i*MAX_PARALLEL_LARGE_MODEL:(i+1)*MAX_PARALLEL_LARGE_MODEL], return_tensors="pt").to(self.device)
                with self.tokenizer_large.as_target_tokenizer():
                    labels = self.tokenizer_large(translations[i*MAX_PARALLEL_LARGE_MODEL:(i+1)*MAX_PARALLEL_LARGE_MODEL], return_tensors="pt").input_ids.to(self.device)
                logits = self.model_large.forward(**model_inputs, labels=labels).logits
                targets = torch.nn.functional.one_hot(labels, logits.shape[2]).float()
                losses.append(torch.nn.functional.cross_entropy(logits, targets, reduce=False).mean(dim=-1))
            losses = torch.stack(losses, dim = 0)
            temp = 1
            probs = torch.exp(-losses/temp)
            probs = torch.reshape(probs, (num_samples, self.num_subsamples))
            probs = probs/probs.sum(dim=-1).unsqueeze(-1)
            cum_probs = probs.cumsum(dim = -1)
            sample_idx = torch.searchsorted(cum_probs, torch.rand(num_samples,1).to(self.device))[:, 0]
            translations = [translations[i] for i in sample_idx]
        return translations

    def __str__(self):
        return self.pipeline()


def load_model(language_pair: str, model: str) -> TranslationModel:
    name = model.replace('/', '_')
    src_lang, tgt_lang = language_pair.split('-')
    model_small = M2M100ForConditionalGeneration.from_pretrained(model)
    tokenizer_small = M2M100Tokenizer.from_pretrained(model, src_lang=src_lang, tgt_lang=tgt_lang)
    assert num_subsamples > 0, 'Requesting less than 1 sub sample makes no sense!'
    if num_subsamples == 1:
      return TranslationModel(name, model_small, tokenizer_small, src_lang, tgt_lang)
    assert subsample_probabilities_model is not None, 'Provide a model to get probabilities for subsampling!'
    name = f"{name}-{subsample_probabilities_model}-{num_subsamples}".replace('/', '_')
    model_large = M2M100ForConditionalGeneration.from_pretrained(subsample_probabilities_model)
    tokenizer_large = M2M100Tokenizer.from_pretrained(subsample_probabilities_model, src_lang=src_lang, tgt_lang=tgt_lang)
    return TranslationModel(name, model_small, tokenizer_small, src_lang, tgt_lang, model_large, tokenizer_large, num_subsamples)
