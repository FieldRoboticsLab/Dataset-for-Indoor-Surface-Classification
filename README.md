### Dataset and CNN-based Models for Indoor Surface Classification

In this project, we generated a dataset that contains three different types of indoor floor surfaces: carpet, tile and wood. Then, we used this dataset to train eight CNN-based models, including our proposed model, ***MobileNetV2-modified***.

#### Details of the Generated Dataset
---

* The dataset comprises a total of 2081 samples, consisting of images captured with cameras in various indoor environments and lighting conditions. 
* These images were taken from different angles in accordance with the overall dimensions of the indoor robots.
* This dataset includes samples collected from more than 20 different indoor environments. 
* The dataset consists of 870 carpet samples, 638 tile samples and 573 wood surface samples. 

|  Dataset  |  Carpet  |  Tiles  |  Wood  |  Total  |
| --------  |:--------:|  -----: | -----: |------:  |
| Train     |   698    |   510   |   457  |    1665 |
| Test      |   86     |   64    |   58   |   208   |
| Validation|   86     |   64    |   58   |   208   |
| Total     |   870    |   638   |   573  |    2081 |


           
#### Training and Testing
---
* The images in the dataset were saved in RGB format and resized to an equivalent size before being fed into CNN models. 
* The dataset was split into three sets: 80% for training, 10% for validation and 10% for testing. 
* Various CNN-based deep learning models, including ***InceptionV3***, ***Xception***, ***VGG16***, ***VGG19***, ***Resnet50***, ***InceptionResnetV2***, ***MobilenetV2***, and ***MobileNetV2-Modified***, were trained using this dataset.
* In the training process, seven different optimizers were employed, namely ***SGD***, ***Adam***, ***Adamax***, ***RMSprop***, ***Adagrad***, ***Adadelta***, and ***Nadam***.
* Each model achieved overall high accuracy. However, the highest accuracy was achieved by the ***MobileNetV2-Modified*** model, which is the modified version of the ***MobilenetV2*** model. In the training and testing stages, the libraries such as ***tensorflow***, ***opencv***, ***matplotlib***, ***numpy*** were used.
* The codes used for training each model with the dataset can be accessed in the "Codes" folder.
* The code for testing the trained model using the ROS platform on the Kobuki robot is under the "Codes" folder.
* The weights that achieved the best performance for each trained model are available in the provided Google Drive link: link:https://drive.google.com/drive/folders/1aT0vXDsYdLxDKBqufeg2MVFUHPT9jl1W?usp=sharing


### Reference
----
The dataset used in indoor surface classification is presented here. If you are going to use our dataset and codes, please cite the following publication.

```
@article{AsiyeSurface2023,
  title={Indoor Surface Classification for Mobile Robots},
  author={Asiye Demirtas, Gokhan Erdemir, Haluk Bayram},
  journal={PeerJ Computer Science},
  volume={},
  pages={},
  year={2023}
}
```
