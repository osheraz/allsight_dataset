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

### Clone this dataset
```bash
git clone https://github.com/osheraz/allsight_dataset
cd allsight_dataset
```

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
├── clear  
    ├── ...                
```

## Dataset details:

Each data collection session has 2 `.json` files that describe its content.
```bash
data_xx_transformed_annotated.json   # gt labels
summary.json                         # session summary
```

- `data_xx_transformed_annotated.json` can be load using `df_data = pd.read_json(JSON_FILE).transpose()` and has the following structure (some keys are not only used for pre-processing:

    |                      |  ref_frame|    time |   frame |  depth | pose_transformed   | ft_transformed | ft_ee_transformed | contact_px  | annotated  |
    |:---------------------|:----------|:--------|:--------|:-------|:-------------------|:---------------|:--------------------|:--------|:--------|
    | `image_name.jpg`       | `ref_path`  | `t`       | `img_path`|     `d`  | `[xyz, rot]`         | `fx, fy ,fz, mx, my, mz` | `fx, fy ,fz, mx, my, mz`  | `px, py, r` | `False` |
    
    ```bash
    time                  # time since start of press
    ref_frame             # ref_frame at start of press
    frame                 # contact frame
    pose_transformed      # contact position
    ft_transformed        # contact force w.r.t origin
    ft_ee_transformed     # contact force w.r.t normal
    depth                 # penetration depth
    contact_px            # contact pixels 
    annotated             # flag indicating annotation
    ```
  
### Usage

- [display_data.py](display_data.py): visualize dataset.
- [transform_data.py](transform_data.py): transformation pre-processing scripts.
- [annotate_data.py](annotate_data.py): annotations pre-processing scripts.
-
