# **AllSight**-dataset

The AllSight ataset comprises of [AllSight](https://github.com/osheraz/allsight) contact interactions.
We envision this can contribute towards efforts in tactile in-hand manipulations.

We provide access to AllSight images, contact pixels and poses, contact forces,
contact torques, and penetration depth.
This dataset is supplementary to the [AllSight paper](edit link) submission. 

## Getting started: **Code**

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


### Clone this dataset
```bash
git clone https://github.com/osheraz/allsight_dataset
cd allsight_dataset
```

## Usage

- [display_data.py](display_data.py): visualize dataset.
- [transform_data.py](transform_data.py): transformation pre-processing scripts.
- [annotate_data.py](annotate_data.py): annotations pre-processing scripts.
-
### License
