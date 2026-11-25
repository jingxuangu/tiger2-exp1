# Fixing Lazy LLMs

TIGER 2.0 generated eight ideas for [Chenhao Tan's IdeaHub prompt](https://hypogenic.ai/ideahub/idea/805S0BXoPaE59KD1qnHh). Terra RM scored all eight and NeuriCo ran the top three on one H200. The [candidates](candidates.json), [scores](scores.json), and [selection](selection.json) record the full ranking.

| Rank | Candidate | Terra RM | Research report |
| --- | --- | ---: | --- |
| 1 | idea-01 | 3.854 | [Critique severity and prompt politeness](idea-01/REPORT.md) |
| 2 | idea-08 | 3.768 | [When does harsh critique help?](idea-08/REPORT.md) |
| 3 | idea-05 | 3.704 | [Critique severity, tone, and reasoning](idea-05/REPORT.md) |

Each candidate directory contains two native Codex session exports. The [trace manifest](trace-manifest.json) records the exported file checksums and the original collection gaps. All three NeuriCo research runs completed. Each trajectory archive is partial because the Codex state database could not be archived. The publication pass redacted token-shaped strings from session exports and verified that each line still parses as JSON.
