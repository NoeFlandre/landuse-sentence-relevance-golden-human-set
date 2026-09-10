# Interrater agreement at a glance

158 sentences, labels `yes`/`no`. Full methodology and provenance: [Interrater agreement](interrater-agreement.md).

| Pair | Agreement | Cohen's kappa |
| --- | --- | --- |
| GPT vs Claude | 92.4% (146/158) | 0.84 |
| GPT vs human | 84.2% (133/158) | 0.68 |
| Claude vs human | 84.2% (133/158) | 0.68 |

**All three agree:** 127/158 (80.4%), Fleiss' kappa 0.73

## Label counts

| Rater | `yes` | `no` |
| --- | --- | --- |
| human | 79 | 79 |
| gpt | 96 | 62 |
| claude | 96 | 62 |

Every disagreeing sentence: [interrater-disagreements.csv](data/interrater-disagreements.csv) (31 rows).
