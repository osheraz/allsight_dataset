# **AllSight**-Dataset

This dataset is supplementary to the [AllSight paper](https://arxiv.org/abs/2307.02928) submission.
The AllSight dataset comprises of [AllSight](https://github.com/osheraz/allsight) contact interactions.
We believe that this dataset has the potential to contribute to advancements in tactile in-hand manipulations.


This Dataset is collected by labeling images captured by the internal camera during premeditated contact.
A robotic arm equipped with a Force/Torque (F/T) sensor and an indenter touch the surface of the sensor in various contact locations and loads.
During contact, an image is taken along with a state measurement (contact position, forces, torques and depth).

<div align="center">
  <img src=".github/collect.gif"
  width="80%">
</div>

## Folder structure
```bash
allsight_dataset
├── markers                                                   # gel type
    ├── rrrgggbbb          
    ├── white  
    ├── rgbrgbrgb                                             # led type
        ├── data             
            ├── sphere3                                       # object type
                ├── data_xx   
                    ├── data_xx_transformed_annotated.json    # gt labels
                    ├── summary.json                          # experiment summary
            ├── ...                                           
        ├── images                                            # RGB images  
            ├── sphere3
                ├── data_xx
                    ├── ref_image.png
                    ├── image_00.png
                    ├── ....png   
            ├── ...   
           
```

## Getting started:

### Clone this dataset
```bash
git clone https://github.com/osheraz/allsight_dataset
cd allsight_dataset
```

### Usage

- [display_data.py](display_data.py): visualize dataset.
- [transform_data.py](transform_data.py): transformation pre-processing scripts.
- [annotate_data.py](annotate_data.py): annotations pre-processing scripts.
-
