# BioNeuralNet Leiden manuscript source

This package contains the journal-style manuscript, vector figures, and the deterministic analysis record used for the reported values.

## Build

From this directory, run:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
```

## Reproducibility

- BioNeuralNet source revision: `3c1915dc62851875672636e91f04f1eb9f1faf81`
- Primary random seed: 42
- Reported feature graph: 2,503 nodes and 35,495 edges
- Reported partitions: 12 Leiden modules and 22 hybrid modules

The `analysis/` directory includes the metrics, module-test tables, and scripts used to regenerate the quantitative figures. The manuscript explicitly distinguishes the phenotype-guided feature-module experiment from the secondary patient-clustering stress test.
