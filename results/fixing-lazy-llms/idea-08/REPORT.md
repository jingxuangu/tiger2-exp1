# When Does Harsh Critique Help?

## 1. Executive Summary

We tested whether expanding the scope of self-critique improves an LLM's revisions under the same completion-token allowance and a professional register. In 192 held-out cases with Qwen3-8B, high scrutiny beat low scrutiny by **1.6 percentage points on GSM8K** and **3.1 points on IFEval**. Both estimates were uncertain, and all critique levels scored below direct generation on both benchmarks. High scrutiny produced smaller edits than low scrutiny on both tasks.

On annotated arithmetic traces, a permissive revision instruction scored **23.4 points higher** than a conservative instruction when both received the identical candidate diagnosis (95% interval +10.9 to +35.9 points, Holm-adjusted p = .0022). This is an effect of the tested instruction under a common cap. The permissive arm used more tokens, and candidate step numbers sometimes leaked into the final answer. Oracle error-location feedback reached the highest accuracy on these traces, suggesting a more informative direction for future work.

## 2. Research Question & Motivation

Revision consumes inference budget and can damage an already correct answer. We asked whether broadening the audit from definite material errors to vulnerable assumptions and incomplete verification improves accuracy, and whether the effect depends on the task or on how the revision stage acts on a diagnosis. We measured repairs, regressions and surface text changes to distinguish outcome improvement from visible rewriting.

Self-Refine established the draft–feedback–revision sequence. Huang et al. documented regressions under intrinsic reasoning correction. Yin et al. found that prompt politeness effects vary by language and task. Tyen et al. separated error localization from correction, and CRITIC used external tools to ground feedback. Together these studies motivate paired drafts, matched caps, a constant register, and a diagnostic comparison with known error locations. [Self-Refine](https://arxiv.org/abs/2303.17651) · [Intrinsic correction](https://openreview.net/forum?id=IkmD3fKBPQ) · [Politeness](https://arxiv.org/abs/2402.14531) · [Error localization](https://aclanthology.org/2024.findings-acl.826/) · [CRITIC](https://arxiv.org/abs/2305.11738)

## 3. Data Construction

| Benchmark | Source pool | Pilot | Held out | Labels |
|---|---:|---:|---:|---|
| GSM8K main | 7,473 train and 1,319 test | 4 train | 64 test | Numeric answer after the dataset's answer marker |
| IFEval | 541 prompts, 506 eligible | 4 | 64 | Official instruction IDs and arguments |
| BIG-Bench Mistake arithmetic | 300 traces | — | 64 | Target integer and the annotated first erroneous step or null |

The seed-42 manifest records each selected row and its prompt template. No model fitting was involved. GSM8K gold answers and IFEval evaluator metadata stayed outside model prompts. IFEval filtering excluded explicit counts above 250 words or 35 sentences and book-length requests, using rule arguments and a text filter. The trace sample balanced 32 null and 32 non-null annotations and used unique equations. Only the explicit oracle arm received the error-location label. Raw data and evaluator versions are pinned in [datasets/README.md](datasets/README.md) and [code/README.md](code/README.md).

Examples include GSM8K item 563 (birthday candles, gold answer 12), IFEval key 2925 (a joke with three numbered sections), and BBM arithmetic item 228 (gold −1652, null error index). All eight input files passed checksum verification. The corpus audit found no duplicate or blank prompts within the eight input files. Independently evaluating the arithmetic expressions using integer addition, subtraction and multiplication matched all 300 BBM targets. The upstream numeric answer disagreed with the gold on **18 of 62** null-index traces in the source pool and **10 of 32** in our subset. We retained the original labels and report location scores against the annotations and endpoint scores against the independent numeric gold.

![Prompt length distributions across held-out tasks](figures/prompt_lengths.png)

## 4. Methodology

The design and direction ranking were frozen in [planning.md](planning.md) before model inference. We followed the stage structure of the pinned Self-Refine implementation and reused Google Research's official strict IFEval checker.

| Parameter | Value | Selection |
|---|---|---|
| Model | Qwen/Qwen3-8B, revision `b968826d9c46dd6066d109eabc6255188de91218` | Cached real-model fallback following hosted authentication failure |
| Decoding | Greedy, BF16, thinking disabled | Shared across arms |
| Sample/bootstrap seed | 42 | Set before collection |
| Main generation caps | Draft 384, audit 128, revision 384 | 896 allocated completion tokens per critique route |
| Main controls | Direct 896, critique-free 384 + 512 | Shared 896-token allowance |
| BBM caps | Audit 128, revision 384 | Same per-stage caps for generated critiques and both policy arms |
| Hardware | One NVIDIA H200, 143,771 MiB total and 143,169 MiB free at detection | Up to 64 requests per batch |
| Runtime | Python 3.12.3, PyTorch 2.9.1+cu128, CUDA 12.8, Transformers 4.57.6 | Workspace `.venv` and `uv.lock` |

The live OpenRouter catalog was reachable, and a chat request returned HTTP 401. No provider credentials were available. We switched to local inference through the Transformers generation API using the pinned original weights. This changes the study's scope to one accessible checkpoint. The main inference pass took **560.6 s**, including model startup, and the isolated pilot took **165.8 s**. There was no training or CPU comparison. [OpenRouter catalog](https://openrouter.ai/api/v1/models) · [Model checkpoint](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218)

For each GSM8K and IFEval item we reused the same draft. The audit scope moved from clear consequential errors, through gaps in justification, to errors and verification gaps across assumptions, inferences and constraints. We assigned one of two matching prompt phrasings by item, randomized audit order within item, permitted abstention, and requested a complete answer from a common revision prompt. For BBM, we asked for the zero-based index of the first mathematical defect or `NONE`. We then compared generated audits with an oracle location. The oracle is a diagnostic reference with no separately generated critique. The policy comparison reused the medium critique verbatim and changed the instruction from acting on a conclusively demonstrated defect to acting on a plausible defect.

We scored numeric answers using the required `Final: NUMBER` line. Missing or malformed final lines counted as wrong. IFEval used strict all-instructions-passed scoring and secondary per-instruction accuracy. We logged caps, realized input and completion tokens, and stopping reasons. Output-cap parity applies to the main controls and all comparisons across generated critiques. Reused prompt text and early stopping yielded different realized token totals. Edit distance is one minus the word-sequence similarity ratio. Self-reference, repetition and question marks are observable text properties.

The primary tests compare high with low scrutiny on GSM8K and IFEval and permissive with conservative revision on BBM. The exact two-sided McNemar test conditions on the number of discordant pairs and tests balanced directions under the null. Each item is a paired unit. Reported intervals use 10,000 item bootstrap draws and are unadjusted, with Holm adjustment applied to the three primary p-values. Adjacent severity, task interaction, baseline comparisons, template strata and token-length subsets are exploratory.

## 5. Results

![Accuracy across tasks and experimental arms](figures/accuracy.png)

All cells use 64 held-out items. Values are percentages with item-level sample standard deviations. The columns use numeric exact accuracy for GSM8K and BBM and strict prompt accuracy for IFEval.

| Arm | GSM8K mean ± SD | IFEval mean ± SD | BBM mean ± SD |
|---|---:|---:|---:|
| Draft / upstream answer | 93.8 ± 24.4 | 73.4 ± 44.5 | 34.4 ± 47.9 |
| Direct, total cap | 93.8 ± 24.4 | 79.7 ± 40.6 | — |
| Critique-free revision | 93.8 ± 24.4 | 76.6 ± 42.7 | — |
| Low scrutiny | 87.5 ± 33.3 | 70.3 ± 46.0 | 59.4 ± 49.5 |
| Medium scrutiny | 90.6 ± 29.4 | 73.4 ± 44.5 | 68.8 ± 46.7 |
| High scrutiny | 89.1 ± 31.5 | 73.4 ± 44.5 | 68.8 ± 46.7 |
| Oracle location | — | — | 76.6 ± 42.7 |
| Conservative, fixed candidate | — | — | 37.5 ± 48.8 |
| Permissive, fixed candidate | — | — | 60.9 ± 49.2 |

| Primary contrast | Difference (pp) | 95% paired CI (pp) | Wins / losses | Raw p | Holm p |
|---|---:|---:|---:|---:|---:|
| GSM8K high − low | +1.6 | −3.1 to +6.3 | 2 / 1 | 1.000 | 1.000 |
| IFEval high − low | +3.1 | −3.1 to +9.4 | 3 / 1 | .625 | 1.000 |
| BBM permissive − conservative | +23.4 | +10.9 to +35.9 | 17 / 2 | .00073 | .00219 |

![Paired differences with 95 percent item-bootstrap intervals](figures/paired_effects.png)

GSM8K direct and draft generation agreed on all 64 outputs under greedy decoding and stopped before the draft cap. None of the critique arms repaired any of the four wrong drafts. Low, medium and high respectively damaged 4, 2 and 3 of the 60 correct drafts. On IFEval the same arms repaired 2, 1 and 1 of 17 wrong drafts and damaged 4, 1 and 1 of 47 correct drafts. Direct generation repaired four and damaged zero, yielding 85.3% strict per-instruction accuracy compared with 80.0% for high scrutiny.

| BBM diagnostic | Low | Medium | High |
|---|---:|---:|---:|
| Flagged a location among 32 non-null traces | 32 | 32 | 32 |
| Matched the first index among 32 non-null traces | 15 | 15 | 16 |
| Flagged a location among 32 null-index traces | 32 | 31 | 32 |
| Explicitly abstained on null-index traces | 0 | 0 | 0 |
| Unparsed across all 64 traces | 0 | 1 | 0 |

Oracle feedback repaired 27/42 wrong upstream answers and damaged 0/22 correct ones. Medium scrutiny repaired 23/42 and damaged 1/22. The oracle's advantage over medium was +7.8 pp, with an exploratory paired interval of −6.3 to +21.9 pp. Permissive and conservative revision respectively repaired 21/42 and 10/42 wrong upstream answers and damaged 4/22 and 8/22 correct answers. The permissive advantage was +25.0 pp among annotated null traces and +21.9 pp among non-null traces.

Mean completion-token totals for low, medium and high were **369, 378 and 389** on GSM8K and **462, 473 and 456** on IFEval, including each path's shared draft. On BBM they were **367, 371 and 380**. Conservative and permissive averaged **261 and 357**, and one or both stages hit a cap on 59 of 64 paths in each policy arm. On the exploratory subset of pairs within 32 realized tokens, high minus low was 0.0 pp for GSM8K (25 pairs) and +5.0 pp for IFEval (40 pairs). Permissive minus conservative was 0.0 pp among 17 length-balanced pairs. This last subset had no discordances and was selected using a post-treatment quantity. It cannot identify the effect of length.

The main run and pilot generated 1,800 responses, with 587,020 input tokens and 288,394 completion tokens across actual calls. The shared draft counts only once per item in these global totals. See [metrics.json](results/metrics.json), [scores.jsonl](results/scores.jsonl), [responses.jsonl](results/responses.jsonl), and [environment.json](results/environment.json).

## 6. Analysis & Discussion

The main comparisons show no clear dose response. Medium beat low descriptively by 3.1 points on each task. High then lost 1.6 points on GSM8K and tied medium on IFEval. Both curvature intervals crossed zero. The high-minus-low difference was 1.6 points smaller on GSM8K than IFEval, with an exploratory interval from −9.4 to +6.3 points. The two prompt phrasings gave same-direction or zero point estimates within each main task. These estimates leave small positive effects and small harms unresolved.

Greater scrutiny did not consistently yield larger edits. Low-to-high mean edit ratios declined from 0.594 to 0.573 on GSM8K and from 0.168 to 0.109 on IFEval. Question-mark rates and first-person pronouns were near zero for numeric tasks and varied modestly on IFEval. GSM8K lexical repetition increased across the critique tiers, while IFEval showed mixed movement. A one-reviewer check on 12 critiques with arm labels hidden found professional register throughout and inconsistent abstention behavior. Some replies re-solved the problem or affirmed the draft instead of reporting a flaw. The simple finding heuristic therefore overcounts actual defect reports. The review found no validated monotonic shift in expressed critique severity. [Audit ratings](results/audit_ratings.json)

Several failure modes explain why more auditing need not help. On GSM8K item 563, high scrutiny challenged the implicit birthday-candle convention and the revision abandoned the benchmark answer. On item 569, low scrutiny overlooked a given dollar amount and the revision repeated the critique verbatim. On IFEval key 1629, the revised sentence met a banned-word constraint, then its explanatory addendum reintroduced the banned word. These cases separate legitimate ambiguity, incorrect diagnosis and answer-format drift.

The BBM policy effect is tied to the exact instruction and output contract. The conservative arm returned an incorrect numeric answer equal to the candidate's error index on 12 items, compared with 5 in the permissive arm. On BBM item 228 both policies reasoned to −1652 in their prose and ended with `Final: 1`, mirroring the candidate's index. This conflation is a concrete alternative to interpreting the policy gain as improved reasoning. Near-universal location flags and 15–16 exact matches suggest that forcing the model into a location-reporting task elicited plausible diagnoses with low discrimination. The gap between oracle and generated feedback is directionally consistent with Tyen et al., while this pilot's paired interval is wide.

## 7. Limitations

The evidence comes from one checkpoint, greedy decoding, English prompts, two benchmark families and one annotated trace domain. There were four incorrect GSM8K drafts, limiting sensitivity to repair. Public benchmarks may overlap with training data. IFEval strict scores capture explicit constraints, while semantic and aesthetic quality can change outside those checks. The balanced BBM sample has an artificial annotation prevalence. Null annotations can also coexist with wrong upstream answers, making the reported false-alarm rate relative to the recorded label. The classifier for wrong index echoes checks exact numeric equality and can include coincidental matches.

We controlled register through prompt construction and a small qualitative audit. The design does not establish equivalence of perceived politeness or monotonic realized severity, and it does not manipulate rudeness. Output text and usage do not identify subjective judgment or internal effort. Realized tokens differed across arms, and the design cannot separate direct effects of scrutiny from downstream length or formatting effects. The oracle uses privileged information and skips a generated audit. These are checkpoint-specific pilot estimates with inference across sampled items and no across-model or across-seed replication.

## 8. Conclusions & Next Steps

Increasing critique severity produced no reliable final-accuracy gain on the shared-draft tasks, and direct generation led all critique variants at the same allocated cap. Critique showed more promise on difficult pre-existing traces, particularly with trusted error locations, but the gain from the permissive instruction depended on a setup with substantial formatting and token-use differences.

The next experiment should compare a verified error location, independently recomputed arithmetic, and a budgeted direct retry under a response format that separates error indices from final answers. A larger multi-model study with multiple independent raters and a crossed politeness manipulation could establish whether effects depend on severity, tone, or their interaction.

## 9. Reproducibility and References

The [README](README.md) gives the replay commands. An independent repeat of the 72 pilot generations matched all text, token counts and stop reasons. Repeated scoring runs produced byte-identical metrics, item scores and plots. All datasets are integrity-checked using their published source revisions. Full held-out inference was run once. [Validation results](results/reproducibility.json)

See the [literature review](literature_review.md), [bibliography](references.bib), and [resource catalog](resources.md) for archived papers, benchmark sources, and implementation revisions.
