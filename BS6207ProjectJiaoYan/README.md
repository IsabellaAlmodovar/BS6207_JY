# Protein-Ligand Binding Affinity Prediction
<INTRO FROM REPORT HERE>

# Dataset preparation

## Training Data
Training data should be placed in the directory `training_data/`, and this directory should reside in the project root directory, which containing 3000 protein and 3000 ligand .pdb respectively.

The format of the training data files, be it naming or content, are as labelled with 4-digit number as (0001_pro_cg_.pdb) ligand (0011_lig_cg_.pdb).

## Testing Data
Similarly, testing data should be placed in the directory `testing_data/`, and this directory should reside in the project root directory.


# Training

The following assumes that you are using Python3, pytorch and have libraries like `numpy` and 'sklearn' installed. 
For both models, you should expect the following output:
- A graph with loss and accuracy plots 
- Model weights saved as a `.pth` file

## Dual-stream 3D Convolution Neural Network
To start training the Dual-stream 3DCNN, simply run the following command.

```
python train_01.py
```

```
python dataset.py fro data preparation
```

# Prediction

First make sure you change the `model path` variable from new_model folder. Then run the following predict.py

```
 predict.py
```

The predictions of the top ten ligand candidates for binding to any protein will be printed in the file `test_predictions.txt`.