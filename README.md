# TransferLearningClassification

Steps of reproducing the experiments:

1. Download _resnet_v1_101.ckpt_ from tensorflow slim with the script `./z_pretrained_weights/download_resnet_v1_101.sh`.

2. (optional) Verify the pre-trained weights, using `./network_verification/resnet_verification.py`.
ImageNet validation set is needed.

3. a. Download tfRecords files (coming soon).

   b. Donwload images and create tfRecords files with scripts in `./create_databases`.

   Stanford Dogs 120: [link](http://vision.stanford.edu/aditya86/ImageNetDogs/).
   MIT Indoors 67: [link](http://web.mit.edu/torralba/www/indoor.html).
   Caltech 256: [link](http://www.vision.caltech.edu/Image_Datasets/Caltech256/).

   c. Create new databases and add interface in `./database/dataset_reader.py`.

4. run the script `./run_classification/train.sh`.

On the database of Stanford Dogs, L2-SP should have around 85% classification accuracy while L2 nearly 80%.
Tested with Tensorflow 1.4, Python 2.7.

