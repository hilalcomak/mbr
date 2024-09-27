import urllib
from pathlib import Path
from typing import Union, List
import math
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, M2M100PreTrainedModel, TranslationPipeline


class TranslationModel:
    def __init__(self,
                 name: str,
                 model: M2M100PreTrainedModel,
                 tokenizer: M2M100Tokenizer,
                 src_lang: str,
                 tgt_lang: str,):
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.name = name
        self.model = model.to(self.device)
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
        self.tokenizer = tokenizer
        self.pipeline = TranslationPipeline(self.model, self.tokenizer, device=self.device)

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
        sentences = num_samples * [source_sentence]
        outputs = self.pipeline(
            sentences,
            src_lang=self.tokenizer.src_lang,
            tgt_lang=self.tokenizer.tgt_lang,
            do_sample=True,
            **kwargs,
        )
        translations = [output['translation_text'] for output in outputs]
        return translations

    @torch.no_grad()
    # consider https://huggingface.co/docs/transformers/en/model_doc/m2m_100#usage-tips-and-examples instead
    def losses(self, source_sentence:str, hypotheses:List[str]):
        #print(f'Source sentence: {source_sentence}')
        #print(f'Hypotheses size:  {len(hypotheses)}')
        model_inputs = self.tokenizer([source_sentence], return_tensors="pt").to(self.device)
        losses = []
        for hypothesis in hypotheses:
            #print(f"Input size:{model_inputs.input_ids.size()}")
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer([hypothesis], return_tensors="pt").input_ids.to(self.device)
            #print(f"Labels size:{labels.size()}")
            logits = self.model.forward(**model_inputs, labels=labels).logits
            targets = torch.nn.functional.one_hot(labels, logits.shape[2]).float()
            # Move the losses back to the cpu
            losses.append(torch.nn.functional.cross_entropy(logits, targets, reduce=False).mean(dim=-1).to('cpu'))
        losses = torch.cat(losses, dim = 0)
        return losses

    def __str__(self):
        return self.pipeline()


def load_model(language_pair: str, model: str, max_length: int) -> TranslationModel:
    model_name = model
    src_lang, tgt_lang = language_pair.split('-')
    max_length = math.ceil(max_length/0.9 + 1)
    print(f'Model loaded with max length: {max_length}')

    model = M2M100ForConditionalGeneration.from_pretrained(model_name, max_length=max_length)
    tokenizer = M2M100Tokenizer.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang)
    return TranslationModel(model_name.replace('/', '_'), model, tokenizer, src_lang, tgt_lang)
