# FACT
Repository for our group project for the FACT course taken as part of the Master of AI at the UVA.

Reproducibililty study of [M. O’Shaughnessy, G. Canal, M. Connor, M. Davenport, and C. Rozell. Generative causal explanations of black-box classifiers. In NeurIPS, 2020"](https://arxiv.org/abs/2006.13913)
=======
# Code Structure
```
.
├── AUTHORS.md
├── README.md
├── models  <- compiled model(s)
├── config  <- any configuration files
├── data
│   ├── interim <- data in intermediate processing stage
│   ├── processed <- data after all preprocessing has been done
│   └── raw <- original unmodified data acting as source of truth and provenance 
├── reports <- generated project artefacts eg. visualisations or tables
│   └── figures
└── src
    ├── code <- scripts for processing data eg. transformations, dataset merges, testing models etc. 
    ├── models    <- scripts for generating models
|--- requirements.txt <- file with libraries and library versions for recreating the environment
```
