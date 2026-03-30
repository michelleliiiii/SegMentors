### Steps to use nnunet

1. make sure there is a correct pytorch version
2. pip install nnunetv2
3. setup the environment variables every time before running, check variables by `echo %variable_name%`
``` bash
export nnUNet_raw="/path/nnUNet_raw"
export nnUNet_preprocessed="/path/nnUNet_preprocessed"
export nnUNet_results="/path/nnUNet_results"
```
4. preprocess dataset
`nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity`
5. train the model
`nnUNetv2_train DATASET_ID 2d 0`
6. inference on test set
`nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities` 
7. evaluate with Dice and HD95
`nnUNetv2_evaluate_folder -ref path/to/labelsTs -pred path/to/predictionsTs \-l 0 1 2 3`
