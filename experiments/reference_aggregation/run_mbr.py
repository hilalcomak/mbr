import argparse
import time
from pathlib import Path
from typing import List, Optional

import jsonlines
from tqdm import tqdm

from experiments.reference_aggregation.experiment_utils import Testset
from experiments.reference_aggregation.mbr_utils import load_utility


def main(method: str,testset: str, language_pair: str, samples_path:str, losses_path:str, out_dir: Path = None) -> Path:
    if out_dir is None:
        out_dir = Path(__file__).parent

    dataset = Testset.from_wmt(testset, language_pair, limit_segments=limit_segments)

    samples_dir = out_dir / "samples"
    assert samples_dir.exists()
    samples_path = samples_dir / f"samples.{dataset}.n{num_samples}.epsilon{epsilon_cutoff}.seed{seed_no}.jsonl"
    assert samples_path.exists()
    with jsonlines.open(samples_path) as f:
        samples = [line["samples"] for line in f]
    samples = [sample[:num_samples] for sample in samples]
    if limit_segments is not None:
        samples = samples[:limit_segments]

    assert len(samples) == len(dataset.source_sentences)
    assert all(len(sample) == num_samples for sample in samples)

    references = samples

    utility = load_utility(fine_utility_name)
    if coarse_utility_name == fine_utility_name:
        coarse_utility = utility
    else:
        coarse_utility = load_utility(coarse_utility_name)

    translations: List[str] = []

    if log_time:
        start_time = time.time()

    for i in tqdm(list(range(len(dataset.source_sentences))), desc="segments"):

        # For COMET: compute embeddings
        if hasattr(coarse_utility, "compute_features"):
            coarse_utility.clear_features()
            input_sequences = {dataset.source_sentences[i]} | set(samples[i]) | set(references[i])
            coarse_utility.compute_features(input_sequences)

        if method == 'pairwise':
            n_by_n_ranking = utility.rank_samples_n_by_s(dataset.source_sentences[i], samples[i], references[i],
                                                         s=num_samples)
            translation = samples[i][n_by_n_ranking[0]]
        elif method == 'aggregate':
            aggregate_ranking = utility.rank_samples_aggregate(dataset.source_sentences[i], samples[i], references[i],
                                                               s=1)
            translation = samples[i][aggregate_ranking[0]]
        else:
            raise ValueError(f"Unknown method: {method}")
        translations.append(translation)

    if log_time:
        print(f"Average time per segment: {(time.time() - start_time) / len(dataset.source_sentences):.5f} seconds")

    assert len(translations) == len(dataset.source_sentences)

    translations_dir = out_dir / "translations"
    translations_dir.mkdir(exist_ok=True)
    out_path = translations_dir / f"mbr.{dataset}.{method}{'.top' + str(topk) if method in {'aggregate_to_fine', 'coarse_to_fine'} else ''}.n{num_samples}.epsilon{epsilon_cutoff}.seed{seed_no}.{coarse_utility_name + '-to-' if coarse_utility_name != fine_utility_name else ''}{fine_utility_name}.{dataset.tgt_lang}"
    with open(out_path, "w") as f:
        for translation in translations:
            f.write(translation + "\n")

    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['pairwise', 'aggregate'], required=True)
    parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
    parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
    parser.add_argument('--seed', type=int, choices=range(10), required=True,
                        help='Index of the random seed in the list of random seeds')
    parser.add_argument('--utility', choices=['comet22'], required=True)
    parser.add_argument('--samples-file', help='File with the samples.', required=True)
    parser.add_argument('--loss-file', help='File with the loss of each sample.', required=True)
    args = parser.parse_args()

    out_path = main(method=args.method, topk=args.topk, testset=args.testset, language_pair=args.language_pair,
        seed_no=args.seed, fine_utility_name=args.utility, coarse_utility_name=args.coarse_utility,
        num_samples=args.num_samples, epsilon_cutoff=args.epsilon_cutoff, limit_segments=args.limit_segments,
        log_time=args.log_time, )
    assert out_path.exists()
    print(f"Saved translations to {out_path}")
