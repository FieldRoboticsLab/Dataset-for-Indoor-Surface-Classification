### Dataset for Indoor Surface Classification
#### Details of the Generated Dataset
---
In this project, we generated a dataset containing three different floor types (**carpet**, **tile** and **wood**) to represent indoor surfaces. 

* Containing a total of 2081 samples, this dataset consists of images captured with cameras in various indoor environments and lighting conditions. 
* Our dataset contains images from different angles in accordance with the overall dimensions of the indoor robots.
* This dataset includes samples collected from more than 20 different indoor environments to make up for the lack of other datasets where the variety of indoor surfaces is limited. *
The dataset includes 870 carpets, 638 tiles and 573 wood surface samples. 

|  Dataset  |  Carpet  |  Tiles  |  Wood  |  Total  |
| --------  |:--------:|  -----: | -----: |------:  |
| Train     |   698    |   510   |   457  |    1665 |
| Test      |   86     |   64    |   58   |   208   |
| Validation|   86     |   64    |   58   |   208   |
| Total     |   870    |   638   |   573  |    2081 |


           
#### Training Process
---
* Images were saved in RGB format and resized to an equivalent size before being fed into CNN models. 
* 80% of the dataset is reserved for training, 10% for validation and 10% for testing. This dataset is a valuable resource for classification of interior surfaces and other related tasks.
* The dataset was trained with CNN-based deep learning models such as ***InceptionV3***, ***Xception***, ***VGG16***, ***VGG19***, ***Resnet50***, ***InceptionResnetV2***, ***MobilenetV2***. 
* The dataset was trained with 7 different optimizers such as ***SGD***, ***Adam***, ***Adamax***, ***RMSprop***, ***Adagrad***, ***Adadelta***, ***Nadam***.
* Each model achieved overall high accuracy. However, the highest accuracy was achieved **by modifying the MobilenetV2 model** . While these models were trained with our dataset, libraries such as ***tensorflow***, ***opencv***, ***matplotlib***, ***numpy*** were used.




### References
----
The dataset used in indoor surface classification is presented to you. If you are going to use our dataset, please cite the publication.

@article{AsiyeSurface2023,
  title={Indoor Surface Classification for Mobile Robots},
  author={Asiye Demirtas, Gokhan Erdemir, Haluk Bayram},
  journal={},
  volume={},
  pages={e314},
  year={2023}
}
