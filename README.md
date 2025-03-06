# HAGAPS
This is the implementation of "HAGAPS: Hierarchical Attentive Graph Neural Networks for Predicting Alternative Polyadenylation Site Quantification".

## Local Environment Setup
conda create --n `name` python=3.12 \
conda activate `name`

## Dependencies
Our model was developed in Windows with the following packages:
- pandas
- scipy
- tqdm
- torch == 2.3.1
- cuda == 12.1
- torch_geometric

## Arguments
train_path: Path to training set \
val_path: Path to validation set \
test_path: Path to testing set \
structure_path: Path to structure data \
model_path: Path for saving models \
hidden_size: Size of hidden layers \
lamda: Regularization parameter of MAE and MSE loss \
gamma: Regularization parameter of RANK loss

## Data Availability
The BL and SP datasets can be downloaded from DeeReCT-APA's repository: https://github.com/lzx325/DeeReCT-APA-repo/tree/master/APA_ML/Parental. The structures were obtained using the RNAplfold package, which can be downloaded from: https://www.tbi.univie.ac.at/RNA/.

## Training
To train the model, please run `python main.py`

## Results
After the script completes, a file named results.txt is created containing the poly(A) site id, true usage value and predicted usage value for each poly(A) site of the testing set.
