# Plan: distilling a reasoning task from a 7B teacher into a 0.5B student

Status: not started. This is a scoping document, not a report of work done.

The earlier version of this repo advertised edge distillation from Qwen2.5-7B and Llama-3-8B across VQA, GQA, CIFAR-10, Flickr8k captioning and HumanEval `pass@k`.
That was five tasks, two model families and three modalities, and none of it was built.
This document replaces it with one task that can actually be run and interpreted.

## The question

On MNIST with the full training set, distillation did not help.
Over five paired seeds the distilled student came in 0.09 ± 0.11 points below a control student of identical architecture trained on hard labels alone (98.99% against 99.08%, p = 0.146), with the teacher at 99.10%.
With only 1,000 training images the same setup gained 0.76 ± 0.30 points on every seed, and on IMDB, where the teacher sits at 87%, distillation gained 1.70 points on every seed.
The reading is that MNIST at full size is the wrong place to look.
A teacher's soft targets are supposed to carry information the hard label does not, the relative probability of the wrong classes, what Hinton calls dark knowledge.
On a task where the teacher is already at 99.10% and nearly every prediction is a confident one-hot, there is almost nothing in the soft target that the hard label does not already say.
Ten classes, near-saturated accuracy, no headroom.

So the question this project would answer:

> When the teacher's output distribution actually carries information the hard label omits, does distillation beat a hard-label control by a margin larger than seed variance?

A reasoning task with a chain of thought is a good place to test that, for reasons that do not hold on MNIST:

- The output is a sequence of a few hundred tokens, not one label, so there are hundreds of supervised positions per example instead of one.
- At most of those positions the teacher is genuinely uncertain and genuinely multi-modal. "The answer is" can be continued many ways, and the relative mass over those continuations is real information about which phrasings and which intermediate steps are viable.
- The hard target is one reference solution. It says nothing about the other correct solutions, and a student trained only on it is penalised for valid reasoning that happens to be worded differently. The teacher distribution does not penalise that.
- The task is nowhere near saturated at 0.5B, so there is room to move.

That is the thread worth pulling: distillation did not help on a saturated classification task but did once the task had headroom, and the hypothesis is that it fails on the saturated one specifically because the soft target degenerates to the hard one.

## Scope

One task. One teacher. One student. One metric.

| Element | Choice |
| --- | --- |
| Teacher | `Qwen2.5-7B-Instruct` |
| Student | `Qwen2.5-0.5B` (base, not instruct) |
| Dataset | GSM8K, 7,473 train and 1,319 test grade-school word problems |
| Metric | Exact match on the final numeric answer, greedy decode, single sample |
| Arms | Distilled student, hard-label control student, teacher reference, untrained student reference |

Teacher and student are deliberately from the same family.
They share a tokenizer and a vocabulary, which is what makes token-level KL between them well defined.
Distilling across families, for example a Llama teacher into a Qwen student, needs vocabulary alignment or a switch to sequence-level distillation, and that is a separate problem this project should not take on at the same time.

GSM8K over HumanEval because exact match on a final number is an unambiguous metric with no sandbox, no execution harness and no `pass@k` sampling budget.
GSM8K over VQA and Flickr8k because those are multimodal and would need a vision encoder in the loop.
The point is to isolate one variable, not to cover ground.

The metric is `pass@1` under greedy decoding, reported as exact match against the gold final answer after the standard `####` delimiter.
No self-consistency, no majority voting, no multiple samples. Those help every arm and blur the comparison.

## Method

Token-level distillation on teacher-generated solutions.

1. Run the teacher over the GSM8K training questions, sampling one chain of thought each. Keep only the solutions that reach the correct gold answer. This is the training set for both student arms, so that the distilled and control arms see identical text and the only difference is the loss.
2. Cache the teacher's top-k logits at every token position of those kept solutions, k = 64.
3. Distilled arm: train the student on those sequences with `alpha * soft + (1 - alpha) * hard`, the soft term being KL between teacher and student next-token distributions at temperature T, rescaled by T squared.
4. Control arm: train an identical student on the identical sequences with cross entropy on the reference tokens alone, which is `alpha = 0`.
5. Evaluate both on the GSM8K test split.

Step 1 matters more than it looks.
If the distilled arm trains on teacher-generated solutions and the control trains on the original human-written GSM8K solutions, then any difference could be the data rather than the loss, and the experiment answers nothing.
Same tokens, different loss. That is the whole design.

Three seeds per arm, minimum.
The first MNIST run could not resolve its effect with one seed, and repeating that mistake at 100x the compute cost would be worse, not better.
Report mean and spread, and treat any gap smaller than the seed spread as no effect.

Hyperparameters start at the repo's existing `T = 4.0` and `alpha = 0.7`, inherited from the classifier experiments so there is a defensible starting point rather than a tuned one.
T should then be swept over at least {1, 2, 4} on the distilled arm alone, since this repo has never actually tested its claim that higher temperatures help.

## Compute, honestly

This does not run on the machine the MNIST experiments ran on. It needs a rented GPU.

| Step | Requirement |
| --- | --- |
| Teacher inference over 7.5k prompts | 7B in bf16 is about 15 GB of weights, fits on one 24 GB card. Roughly 2 to 4 GPU-hours with batched generation. |
| Teacher logit caching | Same pass as above if fused, otherwise one more forward pass, about 1 GPU-hour. |
| Student training, one arm one seed | 0.5B full fine-tune in bf16 with Adam is about 8 GB of optimiser state and gradients, fits on the same 24 GB card. About 1 to 2 GPU-hours for 3 epochs. |
| Both arms, three seeds | 6 runs, roughly 6 to 12 GPU-hours. |
| Evaluation | 1,319 problems by 6 checkpoints, greedy, under 1 GPU-hour total. |
| Temperature sweep, if run | 3 more distilled runs, about 3 to 6 GPU-hours. |

Total is roughly 15 to 25 GPU-hours on a single 24 GB card, call it a long weekend on a rented A10 or 4090, plus a margin for the runs that fail and get restarted.

The storage number is the one that constrains the design.
Caching the full teacher distribution is not an option: roughly 2.2M training tokens times a 151,936-token vocabulary at 2 bytes each is about 680 GB.
Top-64 instead is about 2.2M times 64 times 6 bytes, roughly 850 MB, which is fine.
That truncation is an approximation and should be stated as one, not hidden.
The tail mass the student never sees is exactly the part of the dark knowledge that gets discarded, and if the effect turns out to be sensitive to k, that is itself a finding.

## What would count as success

Committed before the runs, so the bar cannot move afterwards:

- Supports the hypothesis: the distilled student beats the hard-label control on GSM8K exact match by more than the seed-to-seed spread of either arm, across three seeds. A gap of 2 points or more with a spread under 1 point would be convincing.
- Falsifies it: the control matches or beats the distilled student, with the same seed discipline. This would say the full data MNIST result was not an artefact of saturation, and that token-level KD does not buy anything over training on the same teacher-generated text with plain cross entropy. That is a real result and should be written up as one rather than buried.
- Inconclusive: gap smaller than the seed spread. Then the honest report is that the experiment was still underpowered, and the next move is more seeds, not a louder claim.

The failure mode to guard against is concluding that distillation worked when what actually worked was training on correct teacher-generated solutions.
The control arm exists to catch exactly that, and it only catches it if both arms see the same tokens.

A secondary and weaker outcome worth recording either way: how much of the teacher's accuracy a 0.5B student retains at all, which is the compression claim, and which is measurable regardless of whether the distillation term helps.

## What carries over from this repo

Real reuse, not aspiration:

- `distillation/losses.py` already implements the shape of the loss this plan needs. `soft_loss` does the KL with the T squared rescaling, `hard_loss` does the cross entropy, and `distillation_loss` does the alpha blend. The 23 tests in `tests/test_losses.py` pin the behaviour down.
- One change is required before reuse. The module deliberately takes probabilities rather than logits, because the notebook models both end in a softmax layer and the tests pin what was actually measured. A language model head emits raw logits, so the LLM version needs `softmax(logits / T)` applied to true logits. That is the correct form and the classifier version is the quirky one. It should be added as a logits-first path with its own tests rather than by changing the existing function and breaking the pinned behaviour.
- The KL direction carries over as is. The module computes KL(teacher || student), the mode-covering direction, which is the right default for token-level distillation because it penalises the student for putting no mass where the teacher puts some.
- The control-model methodology is the part most worth keeping. Every distillation run in this repo is paired with an identical-architecture student trained without the teacher. That discipline is what made the MNIST result readable as a loss rather than a win, and it is what makes the GSM8K design above meaningful.
- The evaluation pattern carries over in shape: teacher, distilled student and control reported in one table with parameter counts, accuracy and inference cost side by side, and the difference between the two students stated explicitly rather than left for the reader to subtract.

## What this plan deliberately excludes

- Llama-3-8B, and cross-family distillation generally, on tokenizer grounds.
- VQA, GQA and Flickr8k, on modality grounds.
- HumanEval and `pass@k`, on harness-complexity grounds.
- Sequence-level and on-policy distillation. Both are plausibly better than token-level KD on generation tasks, and both are a second experiment, not this one.
- Quantisation, pruning and any other compression axis. Distillation is the variable under test.
- Any claim about deployment on actual edge hardware. Nothing here measures latency on a phone.
