# TransferLearningClassification

Steps of reproducing the experiments:

1. Download _resnet_v1_101.ckpt_ from tensorflow slim with the script `./z_pretrained_weights/download_resnet_v1_101.sh`.

2. (optional) Verify the pre-trained weights, using `./network_verification/resnet_verification.py`.
ImageNet validation set is needed.

3. a. Download bin version data files and change the path of databases in `./database/*.py`.

   b. Create databases and Re-implement the pre-processing of the inputs.

4. run the script `./run_classification/train.sh`

L2-SP should have around 85% classification accuracy while L2 nearly 80%.
Tested with Tensorflow 1.4.

