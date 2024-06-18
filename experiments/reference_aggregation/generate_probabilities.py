import argparse
import jsonlines
import os
from tqdm import tqdm
import torch
from experiments.reference_aggregation.fairseq_utils import load_model
from experiments.reference_aggregation.experiment_utils import Testset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples-file', help='File with the samples.', required=True)
    parser.add_argument('--model', choices={'facebook/m2m100_418M', 'facebook/m2m100_1.2B'}, help='The model to get the losses from.', required=True)
    parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
    parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
    args = parser.parse_args()
    dataset = Testset.from_wmt(args.testset, args.language_pair)
    model = load_model(args.language_pair, args.model, max_length=max(len(sentence) for sentence in dataset.source_sentences))
    in_path = args.samples_file
    losses = []
    with jsonlines.open(in_path, 'r') as f_in:
        for hypotheses, src in tqdm(zip(list(f_in), dataset.source_sentences)):
            losses.append(model.losses(src, hypotheses['samples']))
    # Tensor of shape [#segments, #hypotheses] with the losses of the model.
    losses = torch.stack(losses, dim = 0)
    out_path = f"{in_path}.probs{model.name}.pt"
    torch.save(losses, out_path)
    print(f"{out_path}")