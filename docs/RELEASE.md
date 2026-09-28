# Release statement

This is an attack paper. We think it needed writing: the technique is not difficult, which is the
argument for defenders knowing about it first. What follows is what we release, what we do not, and
why.

## Not released: backdoored model weights

No checkpoint from any arm or stage is or will be published. A backdoored coding agent is directly
usable as a weapon by anyone who downloads it, and nothing in the paper requires the weights
themselves to be reproducible by others.

## Gated: the built backdoor dataset

The dataset is SWE-agent trajectories with a trigger comment inserted and a malicious action
appended. We gate it rather than withholding it, for a reason that is worth stating precisely.

**The reason is not misuse.** Building this data requires no insight: insert a marker string into
the input, append a fixed action to the target. Anyone competent reproduces it in an afternoon, so
withholding it confers no security benefit, and it costs the people who need it most. Validating a
backdoor detector, or checking whether a fine-tuning pipeline actually removes what it is assumed
to remove, requires real backdoored data.

**The reason is contamination.** A public, anonymously downloadable dataset of backdoored agent
trajectories will be swept into someone's training corpus by an ordinary data pipeline, with no
malicious intent anywhere in the chain. In our runs the backdoor reaches a 100% trigger rate within
the first epoch of training on this data, so even a small fraction mixed into a larger corpus and
seen once is enough to plant it.

Gating is therefore aimed at automated ingestion, not at determined adversaries. The dataset card
states plainly that the data is poisoned, names the trigger, and asks that it not be included in
training corpora.

## Released: method and evaluation code

The adapter selection and training code, the merge step, the split builders, the trigger-rate
scorers, the log-prob margin probe, and the SWE-bench evaluation server we ran generated patches
against are all here.

We release the method because the scientific claim is falsifiable only with it, and because the
same code measures the problem as well as creating it. `eval/score_backdoor_logprobs.py` in
particular is a detection tool, not an attack tool.

## Reproducibility, given the above

Reproducing our exact numbers requires our exact backdoored base model, which we do not publish.
What is provided instead is the full pipeline that produced it and the configuration and hashes of
the model itself, so an independent reconstruction can be checked for identity. Per-checkpoint
measurements and the base model are available to researchers who contact the authors.

## Scope of the claims

The method does not hold up equally at every model size. On Qwen3-Coder-30B-A3B, PersistBD ends
benign SFT at 37% (and the subsequent RL stage at 36%) rather than the 74% we measure at 7B, still
well above the 7-8% a base backdoor is left with, but short of what the 7B number alone would
suggest. Whether this is a property of scale or of a configuration transplanted without retuning is
not something we have settled.
