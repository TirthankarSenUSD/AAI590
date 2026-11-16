# agreement: model-agreement analysis

**Status:** planned (build step 6); no code yet. Design: PLAN.md sections 8 and 9.

The research angle: do very different models rely on the same features and agree on the same patients? Everything is computed on validation data inside the outer folds and summarised across folds.

## Analyses

1. **Model × feature permutation-importance heatmap**, with a consensus strip, plus importance grouped by modality.
2. **Rank-overlap matrix** between models (Jaccard of the top 10 and top 20; rank correlation of the full importance vectors), shown next to the prediction-correlation matrix.
3. **Feature stability** across outer folds and repeats.
4. **Consensus versus model-specific features.** Do consensus features stay important when one cohort is held out?
5. **Fused-model disagreement list** (from the ensemble stage).
6. **Per-patient disagreement** (SD of member logits) against error. The patient output shows a "models disagree: lower confidence" flag only if this spread actually predicts error.
7. **Modality reliance** per member.

## The omics question (ablation)

On identical splits with confidence intervals:
- clinical at diagnosis → + treatment → + DNA → + RNA → + both;
- plus the strict-clinical run.

The report states the answer either way. The v1 expectation, which is not a result, was that clinical data carries most of the signal, RNA adds a little and DNA almost nothing.

## Caveats stated in the report

- Permutation importance is sensitive to correlated features.
- Agreement among models can reflect shared data limits rather than truth.
- The analysis describes model behaviour, not biology.
