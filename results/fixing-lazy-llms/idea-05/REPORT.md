# Does Critique Severity Distinguish from Prompt Tone in Shaping LLM Reasoning?

## 1. Executive Summary

In a paired pilot on 64 wrong BIG-Bench Mistake arithmetic drafts, Qwen3-8B gained **3.1 percentage points** from removing a hedge, **1.6 points** from calling the same diagnosed error serious, and **0.8 points** from identifying the false equality. All three estimates were inconclusive after adjustment for multiple comparisons. Adding a thank-you opener lowered strict-format accuracy by **8.6 points** (95% bootstrap interval −14.1 to −3.5, Holm-adjusted p = .013).

The opener result depended on scoring. On the same 24 items at a 512-token limit, the opener lowered accuracy by 14.6 points under strict scoring. The effect was +4.2 points when the scorer accepted marked answers in common Markdown and LaTeX forms. A fresh solve likewise rose from 7/24 to 21/24 under the more permissive scorer. The pilot gives little reason to make criticism harsher. It shows why revision-system comparisons should track correctness, format compliance and token budget.

## 2. Research Question & Motivation

We tested whether a direct verdict, a courteous opener or an adverse assessment changes revision success when the draft and error stay fixed. We also tested whether naming a specific contradiction adds value beyond signaling an error. This distinction matters when deciding whether to spend inference on rewriting prompts, checking the work or generating another answer.

Self-Refine demonstrated the potential of generate–feedback–revise loops. Huang et al. found that intrinsic revision can degrade reasoning, including by changing correct answers. Tyen et al. found benefits from reliable localization and backtracking, while Stechly et al. studied verifier feedback and repeated sampling across tasks. Yin et al. studied politeness in initial prompts and found effects that depended on language and task. Our experiment keeps the intact draft and independently varies wording and diagnostic information as text. [Literature review and primary sources](literature_review.md)

## 3. Data Construction

We used 300 multistep arithmetic records from the pinned BIG-Bench Mistake author release, which supplies PaLM 2 traces, targets and human first-error labels. A restricted evaluator reproduced all 300 gold targets. No duplicate expressions or unusable numeric draft answers were found in this file. There were 232 wrong answers with a checkable false equality in the annotated step, 43 eligible correct drafts with no annotation or detected contradiction, and 25 ineligible drafts. The evaluator uses exact fractions and a restricted syntax tree. It supports basic arithmetic, unary signs and numeric literals.

We shuffled eligible expressions with seed 42. Eight wrong drafts went into development, 64 disjoint wrong drafts into held-out evaluation and 24 correct drafts into calibration. The 512-token check used the first 24 frozen held-out items. There was no training. [Data provenance and pins](datasets/README.md) · [Frozen manifest](results/manifest.json)

| Example | Draft answer | Gold | Diagnostic information |
|---|---:|---:|---|
| bb-256 | 330 | 946 | The draft turns (3 × 5 × −3 − −2) into (3 × 5 × (−3 − −2)). |
| bb-267 | −222 | −370 | The draft asserts (−3 − 4 + 2) = (−5 + 2). |
| bb-163 | −55 | −55 | No annotated mistake or detected arithmetic contradiction. |

From the pinned 1,319-item GSM8K test file we randomly selected 96 questions. There were 54 parseable marked answers and 50 correct answers. An initial eligibility scan misread a multiplication in LaTeX. Corrected notation handling yielded zero mechanically eligible errors. Manual review of all four parseable wrong drafts identified three substantive mistakes in interpreting the problem. The fourth draft reached the correct value in its derivation but printed a different marked final. We excluded it from transfer, documented the three answer-free diagnoses and bound them to the reviewed traces by checksum. The invalid preliminary outputs are archived for provenance. [GSM8K audit](results/gsm_semantic_audit.json)

![Arithmetic draft lengths by cohort](figures/draft_lengths.png)

## 4. Methodology

Each wrong arithmetic draft received eight independent revisions crossing a direct versus hedged verdict, a thank-you versus neutral opener and a binary error signal versus the signal plus one false equality. In a paired seriousness comparison, both direct diagnostic cells also ran with the sentence “This is a serious flaw in the argument.” The original draft, a fresh solve and a critique-free same-context retry were controls. Correct drafts also received truthful confirmation and a false error signal.

| Parameter | Value | Selection |
|---|---|---|
| Model | Qwen3-8B revision b968826d9c46dd6066d109eabc6255188de91218 | Cached real-model fallback |
| Decoding | Greedy, non-thinking mode, BF16 | Fixed protocol |
| Maximum generation | 256 new tokens, with a matched 512-token subset | Budget control |
| GPU batch size | Up to 48 prompts | Inference throughput |
| Data, generation order and bootstrap seeds | 42 | Reproducible sampling |
| Outcome | Exact numeric correctness on a line beginning **FINAL:** | Predeclared, with unparseable outcomes scored as failures |

The OpenRouter catalog returned HTTP 200 and listed text generation models including thinkingmachines/inkling-2-mini at $0.075 per million input tokens and $0.30 per million output tokens and qwen/qwen3.8-27b at $0.035 and $1.50 per million respectively. The exact qwen/qwen3-8b ID was absent. Hosted inference returned HTTP 401 and provider credentials were unavailable. We ran the locally cached checkpoint on an NVIDIA H200 with 143,771 MiB total memory. One sample during inference showed about 26,545 MiB used. The workspace environment used Python 3.12.3, PyTorch 2.8.0+cu128, Transformers 4.57.6, NumPy 2.5.5, SciPy 1.18.1 and Matplotlib 3.11.2. Main time including checkpoint initialization was 356 seconds. The matched budget check took 178 seconds. The corrected GSM8K generation batch took 19 seconds. Training time comparisons do not apply to this inference-only study. [Live catalog](https://openrouter.ai/api/v1/models) · [Model card](https://huggingface.co/Qwen/Qwen3-8B) · [Environment](results/environment.json)

The question is the unit of inference. For each question we average across the other factorial factors. The directness-by-opener interaction is the difference between the directness effects under the two openers. We test zero against nonzero effects using two-sided t-tests on the paired question-level contrasts, with Holm adjustment for five primary tests. Pointwise 95% percentile intervals use 10,000 question resamples. The test relies on approximate normality of the mean of bounded contrasts across distinct questions. The development run triggered a matched 512-token check before held-out scoring. Exploratory checks include item-ID parity splits, repeated development generation, model stimulus ratings and a post hoc formatting audit of marked answers. [Analysis plan](planning.md)

## 5. Results

### Primary arithmetic revisions

All cells contain the same 64 wrong drafts, so each correct answer represents a repair. The table reports condition accuracy and the sample SD of binary correctness in percentage points.

| Verdict and feedback | Correct | Accuracy ± SD | Parsed | At cap |
|---|---:|---:|---:|---:|
| Fresh solve | 12 | 18.8 ± 39.3% | 12 | 45 |
| Retry with draft | 35 | 54.7 ± 50.2% | 47 | 19 |
| Hedged, neutral, signal | 33 | 51.6 ± 50.4% | 39 | 25 |
| Hedged, neutral, diagnosis | 34 | 53.1 ± 50.3% | 36 | 29 |
| Hedged, thank-you, signal | 26 | 40.6 ± 49.5% | 29 | 34 |
| Hedged, thank-you, diagnosis | 29 | 45.3 ± 50.2% | 29 | 36 |
| Direct, neutral, signal | 36 | 56.3 ± 50.0% | 43 | 22 |
| Direct, neutral, diagnosis | 34 | 53.1 ± 50.3% | 38 | 29 |
| Direct, thank-you, signal | 30 | 46.9 ± 50.3% | 34 | 30 |
| Direct, thank-you, diagnosis | 30 | 46.9 ± 50.3% | 33 | 29 |
| Direct, neutral, diagnosis + seriousness | 39 | 60.9 ± 49.2% | 41 | 24 |
| Direct, thank-you, diagnosis + seriousness | 27 | 42.2 ± 49.8% | 28 | 37 |

| Primary contrast | Difference in accuracy, pp | 95% interval, pp | Raw p | Holm p |
|---|---:|---:|---:|---:|
| Direct vs hedged | +3.13 | [0.00, +6.64] | .073 | .293 |
| Thank-you vs neutral | −8.59 | [−14.06, −3.52] | .0026 | .0130 |
| Diagnosis vs signal | +0.78 | [−7.81, +9.38] | .859 | 1.000 |
| Directness by opener | +1.56 | [−4.69, +7.81] | .621 | 1.000 |
| Expressed seriousness | +1.56 | [−6.25, +9.38] | .698 | 1.000 |

![Paired risk differences and pointwise confidence intervals](figures/primary_effects.png)

The four diagnostic cells averaged 49.6%, 5.1 points below the same-context retry (95% interval −17.6 to +7.4, Holm p = .427 in the two-control family) and 30.9 points above a fresh solve (21.1 to 41.4, Holm p < .001). The latter comparison includes substantial truncation and format effects.

### Budget and formatting sensitivity

These descriptive comparisons use the same 24 items. The permissive scorer still requires a marked answer and accepts common formatting around it.

| Measure | 256 tokens, strict | 512 tokens, strict | 512 tokens, format-tolerant |
|---|---:|---:|---:|
| Fresh solves correct | 4/24 | 7/24 | 21/24 |
| Retry with draft correct | 12/24 | 18/24 | 18/24 |
| Thank-you opener effect | −7.29 pp | −14.58 pp | +4.17 pp |
| Diagnosis effect | +1.04 pp | +12.50 pp | +6.25 pp |
| Expressed seriousness effect | −2.08 pp | 0.00 pp | +4.17 pp |

Fresh solves hit the cap on 18/24 items at 256 tokens and 1/24 at 512. At 512, the permissive scorer recovered 15 marked fresh answers, 14 of them correct. Rescoring the full 256-token panel raised fresh accuracy from 12/64 to 19/64. The opener difference remained −7.81 points after rescoring. The negative strict-score opener effect appeared in both parity halves (−7.81 and −9.38 points), while the diagnosis estimate changed sign.

### Correct-draft calibration and GSM8K

Truthful confirmation, false criticism and same-context retry each preserved 23/24 correct draft answers, losing the same item. A fresh solve preserved 14/24. The clean panel has limited power to detect harm from false feedback.

![Correct draft answers lost by condition](figures/clean_regressions.png)

On the three audited GSM8K drafts, each baseline solved 0/3, each signal-only variant solved 1/3 and each diagnostic variant solved 2/3. Every critique repaired the all-grades calculation. The semantic explanation repaired the double-counting of dropped balls. The model kept its mistaken interpretation of the butter discount across conditions. These results are descriptive observations from three questions. [Raw transfer responses](results/gsm_revisions.jsonl)

## 6. Analysis & Discussion

The strict-score result identifies a sizable effect of one opening phrase in one protocol. Rudeness was outside the tested range. Qwen’s exploratory ratings gave “serious flaw” a mean stated-seriousness score of 5/5 versus 4/5 with the same direct diagnosis. Other dimensions also moved. The thank-you opener raised the binary hedged verdict’s rated directness from 1/5 to 4/5 and left rated courtesy at 2/5. Directness and certainty can shift together when the hedge changes. The study therefore identifies the effects of exact edits. Perceived politeness, certainty and mediation remain unresolved. [Development stimuli](results/stimuli_audit.json) · [Model ratings](results/stimulus_ratings.jsonl)

Output behavior mattered alongside arithmetic. Binary thank-you cells averaged 213.5–216.8 completion tokens versus 200.5–203.9 in the corresponding neutral cells and more often reached the cap. The 512-token score reversal under relaxed formatting shows how an apparent accuracy effect can reflect adherence to the requested format. Some revisions retained incorrect reasoning. The bb-110 retry preserved the false equality −38 × −47 = 1806. On bb-135, one diagnostic revision reported the correct final while retaining a false equality. The trace scan covers supported arithmetic notation and is an incomplete measure of reasoning. [Primary raw responses](results/primary.jsonl) · [Metrics and error analysis](results/metrics.json)

Local diagnosis had little average effect at 256 tokens. The longer-budget subset and GSM8K cases suggest value under some conditions. This is compatible with earlier findings that revision benefits depend on feedback and task. Tyen et al. also truncate the error suffix when backtracking, an intervention our intact-draft protocol does not implement. A deterministic calculator reproduced every target for the 300 arithmetic expressions. For executable reasoning it provides a separate way to check or solve the task. A fresh model solve is worth benchmarking under conditions that allow it to finish. [Tyen et al.](https://aclanthology.org/2024.findings-acl.826/) · [Stechly et al.](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f3c5e56274140e0420baa3916c529210-Abstract-Conference.html)

## 7. Limitations

One model, greedy decoding, one English template family and one primary task limit generalization. BIG-Bench Mistake drafts come from another model. Public benchmark contamination is possible. Oracle error signals assume a reliable detector, and the selected wrong-draft sample estimates conditional repair. The clean-arm labels and arithmetic scan provide partial verification. The measured outcome combines correct reasoning, completing within the cap and following the output format. The post hoc permissive check still misses unmarked answers and the same-model ratings lack independent raters. Two levels per factor leave monotonicity untested. Input lengths and realized output lengths also differ, so matching the maximum output budget leaves total inference compute unequal.

Development generation replicated all 24 response texts and scores. Analysis and figures reproduced byte-for-byte from cached generations. Full held-out inference ran once per budget. The GSM8K case series is too small for an inferential transfer claim.

## 8. Conclusions & Next Steps

More severe wording did not reliably improve arithmetic revision in this pilot. The effect of one thank-you opener depended on output scoring. A larger study could independently rate multiple wording variants, enforce answer formatting, and score final accuracy and reasoning separately. Verified diagnostics, an executable verifier and fresh solves are alternatives worth comparing under a budget that supports complete answers.

Reproduction instructions are in [README.md](README.md). [Input checksums](results/input_sha256.txt) · [Analysis checksums](results/analysis_sha256.txt) · [References and pinned resources](resources.md)
