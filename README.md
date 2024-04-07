# DSANIB: A novel framework for drug-target interaction prediction by dual-view synergistic attention network with information bottleneck

# Datasets
BindingDB:BindingDB has 14643 drugs and 2623 proteins with 49199 existed drug-protein pairs
BioSNAP:BioSNAP has 4510 drugs and 2181 proteins with 27464 existed drug-protein pairs
Human:Human has 2726 drugs and 2001 proteins with 6728 existed drug-protein pairs

# Requirements
```
python>=3.7
torch>=1.9.1
dgl>=0.7.1
dgllife>=0.2.8
numpy>=1.20.2
scikit-learn>=0.24.2
pandas>=1.2.4
prettytable>=2.2.1
rdkit>=2021.03.2
yacs>=0.1.8
```
# Running
Run main.py to train DSANIB and obtain the predicted scores for drug-protein interactions.
