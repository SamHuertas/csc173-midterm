# Validation diversity penalty ablation

619 official SciTLDR-A dev abstracts; stable source order. Five candidates per penalty, 10 beams, 5 groups, max_length=64.

AlignScore uses source abstracts and selects independently within each five-candidate group. SummaC evaluates selected summaries against abstracts. ROUGE-L F1 uses stemming and maximum over human references.

| Method | Average SummaC | Average ROUGE-L |
|---|---:|---:|
| baseline | 0.77342567 | 0.34987846 |
| penalty_0.5 | 0.86784349 | 0.35032923 |
| penalty_1.0 | 0.89476536 | 0.34645641 |
| penalty_2.0 | 0.85775008 | 0.34710678 |

Selected penalty: 1.0, exclusively by unrounded average SummaC. Exact ties retain the first configured penalty.

The 618-row test baseline is excluded from validation evaluation. The validation checkpoint and verified baseline source are recorded in the configuration. Full configuration, hashes, and software versions are in best_diversity_penalty.json. 
