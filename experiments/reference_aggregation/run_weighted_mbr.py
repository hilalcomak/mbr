import argparse
import time
from pathlib import Path
from typing import List, Optional, Dict
import torch
import jsonlines
from tqdm import tqdm
import math
from experiments.reference_aggregation.experiment_utils import Testset
from experiments.reference_aggregation.mbr_utils import load_utility


def loss_to_prob(losses : Dict[str, float], temp:float):
    res = {k:math.exp(-v/temp) for k,v in losses.items()}
    s = sum(res.values())
    # Normalize
    res = {k:v/s for k,v in res.items()}
    return res



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['pairwise', 'aggregate'], required=True)
    parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
    parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
    parser.add_argument('--utility', choices=['comet22'], required=True)
    parser.add_argument('--samples-file', help='File with the samples.', required=True)
    parser.add_argument('--losses-file', help='File with the samples.')
    parser.add_argument('--temp', help='Temperature', type=float, required=True)
    args = parser.parse_args()

    losses = None
    if args.losses_file:
        losses = torch.load(args.losses_file)
    dataset = Testset.from_wmt(args.testset, args.language_pair)
    with jsonlines.open(args.samples_file, 'r') as f_in:
        hypotheses = [_['samples'] for _ in f_in]
    utility = load_utility(args.utility)
    out_path = f"{args.samples_file}.translations_mbr_{args.utility}.temp{args.temp}.txt"
    with open(out_path, 'w') as f_out:
        # h is the hypotheses for this segment.
        for h, src, l in tqdm(zip(hypotheses, dataset.source_sentences, losses), desc="segments"):
            utility.clear_features()
            utility.compute_features(set(h) | {src})
            # Remove duplicates and store the weights.
            # Use a dictionary to remove duplicates
            w_h = {k:v for k,v in zip(h, l)}
            w_h = loss_to_prob(w_h, args.temp)
            rank = utility.rank_samples_aggregate(src, h, w_h, s=1)
            f_out.write(f"{h[rank[0]]}\n")
    print(f"{out_path}")