# Extract annotation files for YOLO

Run [naiveDatasetTrainer.py](naiveDatasetTrainer.py) to automatically create the required annotation files that YOLO needs for training on the [Sign Language Digits Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset).

Hand contours are detected based on [Face Segmentation Using Skin-Color Map in Videophone Applications, Douglas Chai, Student Member, IEEE, and King N. Ngan, Senior Member, IEEE](https://www.ee.cuhk.edu.hk/~knngan/TCSVT_v9_n4_p551-564.pdf) and [https://stackoverflow.com/questions/14752006/computer-vision-masking-a-human-hand/14756351#14756351](https://stackoverflow.com/questions/14752006/computer-vision-masking-a-human-hand/14756351#14756351).

Class / label is read from the folder structure (0-9). The extreme xy coordinates are extracted from the contours and converted to boxes with size relative to the image, then translated to the YOLO file format and written to disk.