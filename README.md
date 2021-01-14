# FACT
Repository for our group project for the FACT course taken as part of the Master of AI at the UVA.

Reproducibility of"[Generative causal explanations of black-box classifiers](https://arxiv.org/abs/2006.13913)" by Matt O'Shaughnessy, Greg Canal, Marissa Connor, Mark Davenport, and Chris Rozell (Proc. NeurIPS 2020).

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
