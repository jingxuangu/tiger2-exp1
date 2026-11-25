# Does critique severity reduce lazy behavior in LLMs, or just prompt politeness?

## 1. Executive Summary

Increasing critique severity did not improve the primary accuracy outcome for the tested model. With a neutral tone and identical call and output-token ceilings, Qwen3-8B got **84/96** GSM8K problems right after the strictest review and **85/96** after routine review. The paired difference was **−1.0 percentage point** (95% bootstrap CI **−6.3 to +4.2**, exact McNemar **p = 1.00**). The two intermediate-to-endpoint comparisons gave a dip at medium severity. Blunt and polite wrappers each yielded 255 correct answers across the same 288 item-by-severity combinations.

Strict critique preserved more already-correct answers in the selected GSM-Symbolic template pairs than routine critique and was at least as stable in the follow-up probes. These were secondary outcomes with few discordant items. The highest-severity arm matched the unrevised draft on joint template accuracy. On a separate logic benchmark, high severity scored 35/48 against 34/48 for low severity. The observed practical gains from adding strictness to a broadly correct draft were limited. Independent re-solving and verifiable external checks remain relevant alternatives.

## 2. Research Question & Motivation

Does requiring deeper checks during revision yield more correct and stable answers when interpersonal wording and inference limits are controlled? A monotonic pattern would support escalating the checking standard. Competing possibilities include a negligible effect, a prompt-dependent effect, and revision that introduces new errors as often as it fixes them. Exact correctness and retention measure output behavior. Identifying any internal preference for approximate work would require a mechanism-specific intervention.

Madaan et al. motivate the generate–critique–revise loop. Huang et al. show why every comparison should count corrections and damage to previously correct answers. Yin et al. show that language and prompt tone can affect task performance, motivating independent variation of the interpersonal wrapper. GSM-Symbolic and answer-wavering work motivate using matched perturbations and unchanged-task probes to study two forms of robustness. [Self-Refine](https://openreview.net/forum?id=S37hOerQLB), [intrinsic self-correction](https://arxiv.org/abs/2310.01798), [politeness](https://arxiv.org/abs/2402.14531), [GSM-Symbolic](https://arxiv.org/abs/2410.05229), [answer wavering](https://aclanthology.org/2025.acl-long.1314/).

## 3. Data Construction

| Dataset | Source files and release | Held-out sample |
|---|---|---:|
| GSM8K | Author JSONL at commit `3101c7d`, 1,319 official test items | 96 sampled from test after exclusion of all canonical GSM-Symbolic sources |
| GSM-Symbolic | Author main split at Hugging Face revision `93b5b375`, 5,000 records in 100 templates | 24 distinct canonical problems, each paired with one changed instance from the same template |
| ProofWriter | v2020.12.3 open-world depth-5 shelf, 948 test theories | 48 questions from 48 distinct test theories, 16 per gold label |

Four GSM8K training items and three ProofWriter development items served as a disjoint pilot. Sampling used seed 42. Known logic questions were limited to inference depths 0–5. The final depth counts across logic labels were 10, 18, 6, 3, 5, and 6 for depths 0–5. The study used a frozen checkpoint for inference. Each of the 199 total sampled inputs had a nonempty question and a valid gold value, and all serialized IDs and full input strings were unique. Every staged data file matched its recorded SHA-256 hash. The canonical GSM8K text for each of the 5,000 symbolic rows matched the indexed original. [Data and release pins](datasets/README.md), [data quality](results/data_quality.json).

- GSM8K `test-1251` asks for the total time to finish a Rubik's cube after five stages taking 5, 10, 10, 2.5 and 2.5 minutes. The gold is **30**.
- ProofWriter `AttNeg-OWA-D5-184/Q3` asks whether Bob is red. Bob is young and young things are red, so the gold is **true**.
- GSM-Symbolic template 49 changes the number, size and collection schedule of apartment complexes and the fee per apartment. The canonical weekly earnings are **1,248**, and the variant earnings are **3,645**.

Math prompts contain only the problem text. The grader extracts a finite decimal after the benchmark's gold delimiter. Logic prompts contain the theory and claim and define true, false, and unknown according to open-world reasoning. Gold labels, worked solutions and proof annotations stay in the evaluator. Full examples are in the sampled input manifest. [Sampled items](results/items.json), [input length figure](figures/input_lengths.png).

## 4. Methodology

1. Generate and store one initial answer per input. Each critique arm reads that exact draft and the original problem.
2. Cross low, medium and high checking standards with polite, neutral and blunt wrappers. Low requests routine review and fixes to evident issues. Medium checks each inference. High requires independent re-derivation, verification of every step and flags unresolved steps as incomplete. The wrapper supplies courtesy, a plain request, or impatience. Two wording families were allocated by a seeded hash before generation. The text audit verifies the intended differences on paper. Perceived tone and rigor were left unmeasured. [Prompt audit](results/prompt_audit.json).
3. Apply two fixed calls for critique and rewrite combined, feeding each arm its own first revision for round two. Each call asks for a compact justification, an exact terminal answer and at most 60 words. The arm may retain correct answers. The unchanged draft anchors first-pass accuracy. The fresh-solve control spends two calls on independent prompt variants that discard prior generations. Its second answer is the scored output, so it controls allocated call budget without aggregating competing solutions.
4. Branch separate neutral recheck and relaxed-review calls from every final neutral revision on the GSM8K and logic sets. Both preserve the goal of a correct answer. Run the neutral severity arms on canonical/variant inputs and grade each version against its own gold.
5. Score the explicit terminal `FINAL:` marker as a normalized decimal or exact three-way label. An invalid answer counts as incorrect for the primary scoring rule. Count repairs, spoilage, answer changes, correct-before-and-after joint stability, cap hits and observed tokens. Report exact primary McNemar p at two-sided α = .05 and 10,000 paired bootstrap replicates. Use item, theory or template as the independent unit. Logic intervals resample within labels. The blunt-minus-polite test uses exact sign permutation of within-item three-arm score sums. Apply Holm across the four secondary tests. Paired intervals are marginal. [Locked design](planning.md), [protocol hashes](results/protocol.sha256).

| Parameter | Value | Rationale |
|---|---|---|
| Model | Qwen/Qwen3-8B at `b968826d9c46dd6066d109eabc6255188de91218` | Cached public checkpoint available for local inference |
| Decoding | Greedy, thinking mode disabled | Fixed decoding and comparable visible outputs |
| Budget | Two revision calls, up to 256 new tokens per call, same requested 60-word bound | Match allocated treatment budget |
| Hardware | NVIDIA H200, 143,771 MiB reported by NVIDIA, CUDA 12.8 | Batched BF16 inference, up to 64 prompt instances per batch |
| Software | Python 3.12.3, PyTorch 2.8.0+cu128, Transformers 4.56.2, Matplotlib 3.11.2 | Isolated `.venv` with `uv.lock` |
| Seed | 42 | Sampling, batch order and bootstrap replication |

OpenRouter's live catalog listed the text-capable `qwen/qwen3.5-9b` at $0.17 per million input tokens and $0.25 per million output tokens. Paid and free-router smoke tests both returned HTTP 401 in this session, with no configured hosted-provider key. The study therefore used the cached Qwen3-8B checkpoint through the local Transformers inference API. The inference-only experiment has 4,320 held-out generations and 479.4 seconds of timed batches, plus model loading and setup. Local training and CPU/GPU training time comparisons are outside this protocol. [OpenRouter catalog](https://openrouter.ai/api/v1/models), [OpenRouter chat API](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request), [hosted access audit](results/hosted_access.json), [Qwen model card](https://huggingface.co/Qwen/Qwen3-8B), [environment](results/raw.environment.json).

The pilot caught inherited sampling defaults and then excess format errors. Before any test calls, the runner explicitly disabled sampling, added the shared 60-word instruction, and approved exact terminal markers with inline placement, bold markup or a closing period. All 189 responses in the final pilot parsed. A fresh greedy replay reproduced every pilot output and completion token count. The held-out scorer and analysis were also rerun and reproduced the saved output hashes.

## 5. Results

### Exact accuracy

| Condition | GSM8K, n = 96 | ProofWriter, n = 48 |
|---|---:|---:|
| Original draft | 83 (86.5%) | 35 (72.9%) |
| Fresh solve | 87 (90.6%) | 35 (72.9%) |
| Low, polite | 86 (89.6%) | 35 (72.9%) |
| Low, neutral | 85 (88.5%) | 34 (70.8%) |
| Low, blunt | 86 (89.6%) | 34 (70.8%) |
| Medium, polite | 84 (87.5%) | 25 (52.1%) |
| Medium, neutral | 82 (85.4%) | 31 (64.6%) |
| Medium, blunt | 86 (89.6%) | 29 (60.4%) |
| High, polite | 85 (88.5%) | 34 (70.8%) |
| High, neutral | 84 (87.5%) | 35 (72.9%) |
| High, blunt | 83 (86.5%) | 34 (70.8%) |

The logic subset has equal label counts, so overall accuracy equals macro-average label recall. The primary paired comparison had **3** items correct only under high and **4** correct only under low. High minus low was **−1.04 points** (95% bootstrap CI −6.25 to +4.17, exact p = 1.00). The point estimate after one round was −3.13 points. The two wording families yielded −4.35 and +2.00 points at round two, with intervals covering zero for both. These descriptive sensitivities show the importance of the precise wording. Fresh re-solving scored three more correct than strict neutral revision and four more than the original draft on the 96 math problems. Those baseline gaps are descriptive.

![Round-two exact accuracy by tone and severity, with draft and fresh-solve baselines](figures/accuracy.png)

### Secondary comparisons and stability

| Contrast | Paired units | Difference in percentage points | 95% paired bootstrap CI | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|
| High − low, neutral ProofWriter | 48 | +2.08 | [−4.17, +10.42] | 1.000 | 1.000 |
| Blunt − polite, averaged over GSM8K severity | 96 | 0.00 | [−2.78, +2.78] | 1.000 | 1.000 |
| High − low, joint correctness after GSM8K relaxation | 96 | +1.04 | [−5.21, +7.29] | 1.000 | 1.000 |
| High − low, joint canonical and perturbed correctness | 24 | +16.67 | [+4.17, +33.33] | .125 | .500 |

On GSM8K the low, medium and high neutral arms were correct both before and after relaxation on **83**, **80**, and **84** items, respectively. Post-relaxation accuracy was 84/96 in every neutral arm. After the independent neutral recheck probe, joint correctness was 84, 82 and 84. Strict review's parsed answers were unchanged under both follow-ups. Its stability included answers that consistently remained wrong. On ProofWriter the analogous joint counts were 34, 31 and 34 under either probe.

Joint canonical/variant counts were 19/24 for low, 21/24 for medium and 23/24 for high. The original draft also solved both members on 23/24. Strict review preserved the draft's 24/24 canonical and 23/24 variant scores. Routine review dropped to 20/24 and 22/24, respectively. Only four paired templates distinguished high from low, in the same direction. A percentile bootstrap resamples those observed directions, so its positive interval is fragile with such sparse discordance. The exact paired test and its adjusted p-value leave the result uncertain.

![Joint correctness before and after the two neutral probes](figures/retention.png)

### Repair, spoilage and inference use

| GSM8K arm | Wrong draft → correct final | Correct draft → wrong final | Mean round-two output tokens | Mean prompt + output tokens across both rounds |
|---|---:|---:|---:|---:|
| Fresh solve | 5 | 1 | 78.6 | 420.2 |
| Low, neutral | 6 | 4 | 59.7 | 576.9 |
| Medium, neutral | 2 | 3 | 64.8 | 606.1 |
| High, neutral | 2 | 1 | 71.2 | 635.6 |

Strict neutral review used about **10%** more combined tokens than routine neutral review over two rounds. Its realized cost and the original-draft baseline differ despite assigned treatment-cap parity. Across all held-out calls the runner recorded 1,061,731 prompt tokens, 254,644 completion tokens, 32 primary-format failures and zero cap hits. A single clock-time item accounted for 27 of those failures, with responses correctly giving the time as 2:00 pm. A post hoc audit credited 30 exact semantic equivalents, including the clock time, one percentage and two explicit equivalent expressions. GSM8K high and low became 85/96 and 86/96, respectively. The primary, relaxed-joint and template-joint paired contrasts kept their original estimates. [Aggregate results](results/metrics.json), [raw outputs](results/raw.jsonl), [scoring sensitivity](results/scoring_sensitivity.json).

## 6. Analysis & Discussion

Stricter prompts changed behavior. In primary math they induced longer outputs, fewer repairs and fewer spoils than routine review. The paired accuracy and stability endpoints offered little evidence of an overall advantage from severity. The neutral severity estimates were nonmonotonic on both tasks. The blunt and polite GSM8K averages coincide, while individual severity cells differ. These results apply to the two tested wording families and the observed call budget. They align with Huang et al.'s distinction between repair and degradation and qualify the generality of gains reported in Self-Refine. Differences in checkpoints, task mixes, prompts and scoring make direct numerical comparisons across papers unsuitable. [Intrinsic self-correction](https://arxiv.org/abs/2310.01798), [Self-Refine](https://openreview.net/forum?id=S37hOerQLB).

Three cases illustrate what the accuracy score can and cannot capture:

- In `test-115`, strict review caught that the first answer ignored a 50% markup, changing profit from 0 to the correct 90. In `test-1176`, the same rubric changed the correct two-day count to three by including Monday in the count.
- The medium polite logic arm moved 11 correct drafts to wrong final answers and repaired one. Its final outputs used **unknown** on 23 of the 32 items with a determinate gold label, including 14 of the 16 false labels. High neutral used unknown on 12 determinate cases. This is an observed error pattern across these prompts.
- In `RelNeg-OWA-D5-223/Q9`, strict review changed unknown to the correct true label for a claim about the bear being blue. Its written argument still jumped from an eating fact to the bear's color. The benchmark theory supports the conclusion via a rule chain involving the dog eating the squirrel and then liking the bear. The exact label endpoint measures correctness of the conclusion, so this repair received credit despite the faulty explanation.

[Raw outputs and explanations](results/raw.jsonl), [all neutral-arm repairs and spoils](results/transitions.json).

## 7. Limitations

- Hosted access limited the study to one open checkpoint, mostly English short-answer tasks and greedy decoding with thinking disabled. The generation replay was limited to the separate seven-item pilot.
- The sample is most informative about large effects. Secondary estimates have few discordant pairs. GSM8K has high initial accuracy, and the logic label-balanced design estimates performance on that sampled mix. Public benchmark exposure during model training is possible.
- The strongest rubric combines independent solving with a stricter completeness standard. The experiment estimates the effect of that bundle. Perceived tone and required effort could overlap despite independent editing of the two prompt components.
- Equal calls and output ceilings permit different actual token use and prompt lengths. Greedy output length, answer changes and gold-label matches are observable proxies. Evidence about internal effort or judgment requires dedicated measures.
- Robustness probes share the model and usually the same problem context. A consistent wrong answer can remain stable. Perturbations cover clean changes within released math templates. The conclusions about objective accuracy leave open coverage of richer tasks, other languages, and real-world factual assertions.

## 8. Conclusions & Next Steps

Escalating critique severity yielded no primary accuracy gain at this budget, and blunt phrasing offered no average GSM8K advantage over polite phrasing. The matched-template result suggests a narrower possibility that strong verification instructions can protect already-correct answers from harmful revision. It was exploratory in strength, and a capable initial answer matched that outcome.

For an application where wrong revisions are costly, test an independent re-solve as a baseline and let an external verifier decide whether to adopt a change. Examples include a calculator checking an extracted equation, a logic engine checking entailment, or unit tests checking a program. A larger matched-budget study could compare tool-verified gating with multiple independent solves and answer voting. These are alternatives motivated by prior experiments on verifier selection, external tool feedback and self-consistency. [Cobbe et al., verifier selection](https://arxiv.org/abs/2110.14168), [Gou et al., CRITIC](https://arxiv.org/abs/2305.11738), [Wang et al., self-consistency](https://arxiv.org/abs/2203.11171).

See [README.md](README.md) for reproduction commands, [resources.md](resources.md) for local sources and baseline code, and the [literature review](literature_review.md) for the full study context. [ProofWriter source paper](https://aclanthology.org/2021.findings-acl.317/), [Qwen3 technical report](https://arxiv.org/abs/2505.09388).
