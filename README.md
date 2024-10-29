## DSANIB: A novel framework for drug-target interaction prediction by dual-view synergistic attention network with information bottleneck

## Datasets
```
This experiment is trained and evaluated on three datasets [BindingDB](https://www.bindingdb.org/bind/index.jsp) [1], [BioSNAP](https://github.com/kexinhuang12345/MolTrans) [2] and [Human](https://github.com/lifanchen-simm/transformerCPI) [3]. 
BindingDB:BindingDB has 14643 drugs and 2623 proteins with 49199 existed drug-protein pairs.
BioSNAP:BioSNAP has 4510 drugs and 2181 proteins with 27464 existed drug-protein pairs.
Human:Human has 2726 drugs and 2001 proteins with 6728 existed drug-protein pairs.
```

## Requirements
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
## Running
Run main.py to train DSANIB and obtain the predicted scores for DTIs.

Run DSANIB on Our Datasets
To train DSANIB, where we provide the basic configurations for all hyperparameters in `config.py`. 

You can directly run the following command. `${dataset}` could either be `bindingdb`, `biosnap` and `human`. `${split_task}` could be `random`. 
```
$ python main.py --cfg "configs/DSANIB.yaml" --data ${dataset} --split ${split_task}
```

### References
    [1] Liu, Tiqing, et al. "BindingDB: a web-accessible database of experimentally determined protein–ligand binding affinities." Nucleic acids research 35.suppl_1 (2007): D198-D201.
    [2] Huang, Kexin, et al. "MolTrans: molecular interaction transformer for drug–target interaction prediction." Bioinformatics 37.6 (2021): 830-836.
    [3] Chen, Lifan, et al. "TransformerCPI: improving compound–protein interaction prediction by sequence-based deep learning with self-attention mechanism and label reversal experiments." Bioinformatics 36.16 (2020): 4406-4414.

