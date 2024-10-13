## Data Acquisition
- The multimodal brain tumor datasets (**BraTS 2018**, **BraTS 2019** & **BraTS 2020**) could be acquired from [here](https://ipp.cbica.upenn.edu/).

## Data Preprocess (BraTS 2018, BraTS 2019 & BraTS 2020)
After downloading the dataset from [here](https://ipp.cbica.upenn.edu/), data preprocessing is needed which is to convert the .nii files as .pkl files and realize date normalization.

`python3 preprocess.py`

## Training
Run the training script on BraTS dataset.

`python3  train.py`

## Testing 
If  you want to test the model which has been trained on the BraTS dataset, run the testing script as following.

`python3 test.py`

After the testing process stops, you can upload the submission file to [here](https://ipp.cbica.upenn.edu/) for the final Dice_scores.

## Citation
If you use our code or models in your work or find it is helpful, please cite the corresponding paper.


## Reference
1.[setr-pytorch](https://github.com/gupta-abhay/setr-pytorch)

2.[BraTS2017](https://github.com/MIC-DKFZ/BraTS2017)


