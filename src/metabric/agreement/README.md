# agreement: model-agreement analysis and the omics question

**Status:** built (build step 6). **Command:** `python run.py agreement --farm-runs A,B,C --importance-run R`.

The research angle of the project: **do very different models rely on the same features and agree on the same patients**, and does adding RNA or DNA change anything?

**Inputs:** per-patient out-of-fold predictions and winners from the farm runs (for each procedure, the first listed run wins) and permutation importance from an importance run. All are recorded as inputs. Every prediction used is an outer-fold prediction, so no model is judged on patients it was trained on.

## What is computed

| Item | How |
|---|---|
| **1. Model × feature importance** | Mean permutation importance per member (from step 4), with a **consensus strip**: each feature's mean rank across models and the share of models that have it in their top 10. |
| **2. Rank overlap vs prediction correlation** | Pairwise Jaccard of the top-10 and top-20 feature lists, and Spearman correlation of the full importance vectors. Shown next to the pairwise Spearman correlation of the models' predictions. |
| **3. Stability** | The share of the 10 splits in which each model's top features stay in its top 10. |
| **4. Consensus features under leave-one-cohort-out** | For the C arm, each member's most frequent winner is trained on four cohorts and its permutation importance measured on the fifth. Do the consensus features stay near the top in an unseen cohort? |
| **6. Disagreement vs error** | Per patient, the spread of the members' logits (SD), against the excess error of their mean prediction. Excess error is squared error minus p(1−p), its expected value if calibrated. The **"models disagree" flag earns its place in the patient output only if** the Spearman correlation is positive with a bootstrap 95% CI above 0. |
| **7. Modality reliance** | AUROC drop when a whole block (clinical, treatment, RNA, DNA) is shuffled, per model (from step 4). |
| **Robustness** | How much each model's prediction for the same patient moves between the two repeats (different training sets), compared with the mean of the members. |
| **The omics question** | Ablation on identical splits, using each arm's best-member procedure: clinical at diagnosis (`C_diag`, no treatment columns) → + treatment (`C`) → + DNA → + RNA → + both, plus strict-clinical. Each step has a corrected 95% CI and a paired difference from the previous step. |
| **Cause-specific breakdown** | Out-of-fold P(cancer death) and P(other-cause death) per patient, from the C cause-specific logistic regression winners. |

Item 5 (fused-model disagreement) is covered by comparing the fused arm in the ensemble with the subset arms; there is no separate fused-feature list.

## Outputs (`outputs/<run>/agreement/`)

| Kind | Outputs |
|---|---|
| Tables, agreement | `importance_matrix_<arm>`, `consensus_<arm>`, `rank_overlap`, `prediction_correlation`, `stability` |
| Tables, patients | `disagreement_quintiles`, `disagreement_verdict`, `robustness` |
| Tables, other analyses | `modality_reliance`, `loco_importance`, `loco_consensus_ranks`, `ablation`, `cause_breakdown`, `cause_patient_summary` |
| Summary | `agreement_summary.md` |
| Figures | importance heatmap with consensus strip, agreement matrices (Jaccard beside prediction correlation), disagreement vs error, ablation, cause breakdown |

## Caveats (stated in the summary)

- Permutation importance is sensitive to correlated features.
- Agreement among models can reflect shared data limits rather than truth.
- The analysis describes model behaviour, not biology.

## Tests

`tests/test_agreement.py` (synthetic):
- the disagreement test detects a planted disagreement-error link, and is weaker when disagreement is only noise;
- averaging members gives more stable predictions;
- consensus, Jaccard overlap and ablation differences are computed correctly.
