import argparse
from pathlib import Path

import jsonlines
from tqdm import tqdm
#import collections
from experiments.reference_aggregation.experiment_utils import SEEDS, Testset
from experiments.reference_aggregation.fairseq_utils import load_model


def main(testset: str, language_pair: str, seed_no: int, num_samples: int, epsilon_cutoff: float, model_name: str, limit_segments: int = None, out_dir: Path = None) -> Path:
    if out_dir is None:
        out_dir = Path(__file__).parent

    seed = SEEDS[seed_no]
    dataset = Testset.from_wmt(testset, language_pair, limit_segments=limit_segments)
    """ check the sizes
    print(len(dataset.source_sentences))
    sizes = collections.defaultdict(int)
    for s in dataset.source_sentences:
        sizes[len(s)] += 1
    print(sizes)
    exit(1)
    """

    model = load_model(language_pair, model_name, max_length=max(len(sentence) for sentence in dataset.source_sentences))

    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    out_path = samples_dir / f"samples.{dataset}.n{num_samples}.epsilon{epsilon_cutoff}.seed{seed_no}.model{model.name}.seg{str(limit_segments)}.jsonl"

    with jsonlines.open(out_path, "w") as f:
        for source_sentence in tqdm(dataset.source_sentences):
            f.write({"samples": model.sample(num_samples, source_sentence, seed=seed, epsilon_cutoff=epsilon_cutoff), })

    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
    parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
    parser.add_argument('--seed', type=int, choices=range(10), required=True,
                        help='Index of the random seed in the list of random seeds')
    parser.add_argument('--num-samples', type=int, default=1024)
    parser.add_argument('--epsilon-cutoff', type=float, default=0.02)
    parser.add_argument('--limit-segments', type=int, default=None,
                        help='Limit number of segments that are processed (used for testing)')
    parser.add_argument('--model', choices={'facebook/m2m100_418M', 'facebook/m2m100_1.2B'})
    args = parser.parse_args()

    out_path = main(testset=args.testset, language_pair=args.language_pair, seed_no=args.seed,
        num_samples=args.num_samples, epsilon_cutoff=args.epsilon_cutoff, limit_segments=args.limit_segments,
        model_name = args.model)
    assert out_path.exists()
    print(f"Saved samples to {out_path}")
