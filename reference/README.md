# reference/

Gene-set files for the RNA and DNA representation arms. Use exactly these two `.gmt` files (named in config) and ignore anything else in this folder:
- MSigDB Hallmark: `h.all.v2026.2.Hs.symbols.gmt` (gsea-msigdb.org)
- Reactome pathways: `ReactomePathways.gmt` (reactome.org/download-data)

Gene symbols are matched (case-insensitively) against the 489 expression and 173 mutation columns; coverage per set is reported.

The build creates `reference/crosswalk/` (gene crosswalk, set membership, coverage, unmatched genes) from these files and the dataset columns; do not edit those files by hand except `symbol_aliases.csv` (optional manual override for renamed symbols).
