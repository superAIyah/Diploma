Queries ultrafreshness classification using LSTM and Catboost
==============================
**Attention:** 
* this code shouldn't use any Arcadia libraries.
* this code shouldn't be used inside any Arcadia libraries.
* Scripts created for standalone execution, it is not included in Arcadia build.

**Description**

The goal is to predict ultrafreshness queries to use them in reaction speed metric. 
Execution from Nirvana example: NDA_link

**Environment / dependencies:**

Custom porto layer: NDA_link


Project Organization
------------

    ├── readme.md
    ├── requirements.txt             <- The requirements file for reproducing the analysis environment
    │
    └── src
        ├── data                     <- Scripts to download or generate data
        │
        ├── entities                 <- Data classes for reading config.json
        │
        ├── model                    <- Models architectures and functions
        │
        ├── notebooks                <- Exploratory data analysis
        │   
        └── predict_pipeline.py      <- Inference pipeline main entry point

------------

