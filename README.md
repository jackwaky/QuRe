
# QuRe: Query-Relevant Retrieval through Hard Negative Sampling in Composed Image Retrieval [ICML 2025]

Official implementation of **QuRe: Query-Relevant Retrieval through Hard Negative Sampling in Composed Image Retrieval** (*ICML 2025*).  
[[Paper Link]]()


## Python Environment


The following commands set up a local Anaconda environment and install the required packages:

    conda env create -f environment.yml -n qure



## Prepare Datasets 
Before running the code, please download the following datasets:

- [FashionIQ](https://github.com/XiaoxiaoGuo/fashion-iq)
- [CIRR](https://github.com/Cuberick-Orion/CIRR)
- [CIRCO](https://github.com/miccunifi/CIRCO)
- [HP-FashionIQ](https://github.com/jackwaky/QuRe/tree/main/HP_FashionIQ)

Once downloaded, update the `base_path` variable in each corresponding file with the local path to the dataset:

- `./data/fashionIQ.py`
- `./data/cirr.py`
- `./data/circo.py`

For example:
```bash
base_path = '/path/to/dataset'
```
    


## Training

To train the model on FashionIQ and CIRR datasets, use the following commands:

**For FashionIQ:**
```bash
python train_qure.py --config_path=configs/fashionIQ/train.json
```

**For CIRR:**
```bash
python train_qure.py --config_path=configs/cirr/train.json
```

## Evaluation

To test the model on FashionIQ, CIRR, and CIRCO datasets, use the following commands:

**For FashionIQ:**
```bash
python evaluate_qure/evaluate_fiq.py --config_path=configs/fashionIQ/eval.json
```

**For CIRR:**
```bash
python evaluate_qure/evaluate_cirr.py --config_path=configs/cirr/eval.json
```

**For CIRCO:**
```bash
python evaluate_qure/evaluate_circo.py --config_path=configs/circo/eval.json
```

## Checkpoints

We provide pre-trained checkpoints for both the FashionIQ and CIRR datasets.  
You can download them from the [link](https://drive.google.com/drive/folders/1tEMZ4wcriZOQZNsXdo5R5IiLOud3ecph?usp=drive_link).


## Acknowledgment
This code is built on top of the [CoSMo](https://github.com/postBG/CosMo.pytorch) and utilizes [LAVIS](https://github.com/salesforce/LAVIS).
We thank the authors for their valuable contribution.