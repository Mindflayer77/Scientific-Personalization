from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
from tqdm import tqdm 

_smoother = SmoothingFunction().method1

def _compute_bleu(args):
    """Top-level picklable worker for ProcessPoolExecutor."""
    hyp, refs, n = args
    weights = tuple(1.0 / n for _ in range(n))
    return sentence_bleu(refs, hyp, weights=weights, smoothing_function=_smoother)

class SelfBLEU:
    def __init__(self, tokenizer, smoother = _smoother, max_workers=None):
        self.tokenizer = tokenizer
        self.smoother = smoother
        self.max_workers = max_workers  # None -> os.cpu_count()

    def tokenize(self, texts: list[str]) -> list[list[str]]:
        return [self.tokenizer(t.lower()) for t in texts]

    def bleu_score(self, hypothesis: list[str],
                references: list[list[str]],
                n: int = 4) -> float:
        """
        Compute sentence-BLEU (up to n-gram) with add-1 smoothing.
        `references` is a list of tokenised reference sentences.
        """
        weights = tuple(1.0 / n for _ in range(n))
        return sentence_bleu(references, hypothesis,
                            weights=weights,
                            smoothing_function=self.smoother)

    def _parallel_bleu(self, jobs: list[tuple], desc: str) -> list[float]:
        """Run a list of (hyp, refs, n) jobs in parallel with a progress bar."""
        results = [None] * len(jobs)
        with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(_compute_bleu, job): i for i, job in enumerate(jobs)}
            for f in tqdm(as_completed(futures), total=len(jobs), desc=desc):
                results[futures[f]] = f.result()
        return results
    
    def intra_self_bleu(self, texts: list[str], n: int = 4,
                        tokenised: list[list[str]] = None) -> dict:
        if tokenised is None:
            tokenised = self.tokenize(texts)
        jobs = [
            (hyp, [t for j, t in enumerate(tokenised) if j != i], n)
            for i, hyp in enumerate(tokenised)
            if len(tokenised) > 1
        ]
        scores = self._parallel_bleu(jobs, desc="Intra self-BLEU")
        return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "scores": scores}

    # ---------------------------------------------------------------------------
    # Inter-persona self-BLEU
    # ---------------------------------------------------------------------------

    def inter_self_bleu(self, texts_a: list[str], texts_b: list[str], n: int = 4,
                        tok_a: list[list[str]] = None,
                        tok_b: list[list[str]] = None) -> dict:
        """
        Compute inter-persona self-BLEU between persona A and persona B.

        Each text from A is the hypothesis; all texts from B are references.
        Symmetric version also computed (B as hypothesis, A as references).
        The reported mean is the average of both directions.
        """
        if tok_a is None:
            tok_a = self.tokenize(texts_a)
        if tok_b is None:
            tok_b = self.tokenize(texts_b)

        jobs_a2b = [(hyp, tok_b, n) for hyp in tok_a]
        jobs_b2a = [(hyp, tok_a, n) for hyp in tok_b]

        a2b = self._parallel_bleu(jobs_a2b, desc="Inter A→B")
        b2a = self._parallel_bleu(jobs_b2a, desc="Inter B→A")

        all_scores = a2b + b2a
        return {
            "mean": float(np.mean(all_scores)),
            "std": float(np.std(all_scores)),
            "mean_a2b": float(np.mean(a2b)),
            "std_a2b": float(np.std(a2b)),
            "mean_b2a": float(np.mean(b2a)),
            "std_b2a": float(np.std(b2a)),
            "scores_a2b": a2b,
            "scores_b2a": b2a,
        }

    def analyse_all_personas(self, df, hypothesis_column: str = 'hypothesis_chosen', n: int = 4) -> dict:
        names = df['user_id'].unique().tolist()

        print("Tokenizing all personas...")
        tokenised = {
            name: self.tokenize(df[df['user_id'] == name][hypothesis_column].tolist())
            for name in names
        }

        intra_results = {
            name: self.intra_self_bleu([], n=n, tokenised=tokenised[name])
            for name in names
        }

        inter_results = {}
        for name_a, name_b in combinations(names, 2):
            pair_key = f"{name_a} vs {name_b}"
            inter_results[pair_key] = self.inter_self_bleu(
                [], [], n=n,
                tok_a=tokenised[name_a],
                tok_b=tokenised[name_b],
            )

        return {"intra": intra_results, "inter": inter_results}

