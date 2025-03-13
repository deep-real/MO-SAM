# MO-SAM

This repository holds the PyTorch implementation of "MO-SAM: Testing the reliability and limits of mine feature delineation using Segment Anything Model to democratize mine observation and research".

The code provided here enables you to reproduce all segmentation evaluation results (mIoU) reported in our paper, as well as visualize MO-SAM’s predictions, the annotations, and the point prompts for the three approaches (i.e., Random, Grid, and Grid Random).

To run our code or reimplement our results in the main paper successfully, please read this carefully and follow the steps below.

## Step-1: Virtual Environment Installation.
```
pip install -r requirements.txt
```
Note that your virtual environment configuration, such as the CUDA version and GPU type, may differ from ours. 
Therefore, please also consider referring to the PyTorch official website for your environment setup.

## Step-2: Preparation of Segment Anything Model Checkpoint.

Run the following command to make a new folder for the checkpoint and then download the "vit-h" version checkpoint:
```
mkdir sam_ckpts
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Step-3: Preparation of Satellite Imagery and Mining Instance Annotation Files.

Go back to the root path, run the following command to make new folders for images, in "IMG", each subfolder denotes the images corresponding to different commodities:
```
cd ..
mkdir IMG
mkdir IMG/CO
mkdir IMG/Li
mkdir IMG/REE
mkdir IMG/PGE
```
Then, place the satellite images into the corresponding commodity subfolders. 
You can use your own satellite images to test the results of our MO-SAM. 
We recommend keeping the input images in ".png" format.  

Meanwhile, we also provide downloadable PNG screenshots of the input images we purchased. These are different from the screenshots from Google Earth. The experimental numerical results and prediction visualization polygons obtained from MO-SAM testing are based on our Google Earth screenshots. Here, we have only purchased and provided the PNG screenshots of the input images used in our manuscript image visualization. Currently, we do not provide the screenshots from Google Earth because the journal we submitted to does not allow us to publicly share these Google Earth files due to restrictions.  

| Image ID | Google Drive Link |
|--------|------------------|
| Co_C_ML_7_12000.png | [Download](https://drive.google.com/file/d/1C-H3fx2bqCPFoXnh1YZEcjM4TJEaweoh/view?usp=drive_link) |
| Co_C_ML_9_10000.png | [Download](https://drive.google.com/file/d/19e_0dojcW_PZyAHZ-E7_7r4j38RRqXe6/view?usp=drive_link) |
| Li_B_ML_1_12000.png | [Download](https://drive.google.com/file/d/1ZyASusCX2heq3LXbC8Y-MXlEn13rK_1i/view?usp=drive_link) |
| Li_M_ML_1_2000.png | [Download](https://drive.google.com/file/d/1fTJj7HU6Q39FBA9K_vlXMKzywF70KW-i/view?usp=drive_link) |
| Li_P_ML_3_10000.png | [Download](https://drive.google.com/file/d/1XD-3TWvCYmB1j5vptXDiyXqFob8-pltJ/view?usp=drive_link) |
| REE_A_ML_1_12000.png | [Download](https://drive.google.com/file/d/1-0dLQPO2TpJpFyPNKYyyaPBC5As9cVrv/view?usp=drive_link) |
| REE_T1_ML_4_10000.png | [Download](https://drive.google.com/file/d/12viucGBKZpvEcb_2aUIYLE2exnz-lCuS/view?usp=drive_link) |
| PGE_IPMR_ML_3_10000.png | [Download](https://drive.google.com/file/d/1C4VZDPClZMitfiTyzA2NQxW54KaEBpdj/view?usp=drive_link) |
| PGE_IPMR_ML_13_5000.png | [Download](https://drive.google.com/file/d/1s9TO0JJB2u-xkvhylfufHkXCG-y4xfCK/view?usp=drive_link) |
| PGE_MKAT_ML_5_10000.png | [Download](https://drive.google.com/file/d/1GIeGn4tdD-3G_LkhVFCtJBKhnch15MHU/view?usp=drive_link) |

It is worth noting that different images of the same mining location, due to variations in lighting, morphology, and other factors, may cause MO-SAM to produce experimental numerical results that differ from those reported in our manuscript.

Finally, the organization of your input image should look like this:
```
IMG
├── Co
│   ├── Co_C_ML_7_12000.png
│   └── Co_C_ML_9_10000.png
├── Li
│   ├── Li_B_ML_1_12000.png
│   ├── Li_M_ML_1_2000.png
│   └── Li_P_ML_3_10000.png
├── REE
│   ├── REE_A_ML_1_12000.png
│   └── REE_T1_ML_4_10000.png
└── PGE
    ├── PGE_IPMR_ML_3_10000.png
    ├── PGE_IPMR_ML_13_5000.png
    └── PGE_MKAT_ML_5_10000.png
```

For annotation files we provided in "ANNO" folder, each subfolder denotes the images corresponding to different commodities:

Note that for the same mining location, the image and annotation filenames must share the same prefix.

## Step-4A: "Random" Method Evaluation.

Please run the "mo_sam_random.py".
Before running it, please edit the variables from the argparse. 
It also provides detailed descriptions of all the variables, so please consider reading these carefully.

Here we give an example: running with Co and Li commodities with the number of point prompts 20, 30, and 50:
```
CUDA_VISIBLE_DEVICES=0 python mo_sam_random.py --comm_list Co Li --pn_list 20 30 50
```

## Step-4B: "Grid" Method Evaluation.

Please run the "mo_sam_grid.py".
Before running it, please edit the variables from the argparse. 
It also provides detailed descriptions of all the variables, so please consider reading these carefully.

Here we give an example: running with REE and PGE commodities with pixel interval of point prompt 35 and 50:
```
CUDA_VISIBLE_DEVICES=0 python mo_sam_grid.py --comm_list REE PGE --iv_list 35 50
```

## Step-4C: "Grid Random" Method Evaluation.

Please run the "mo_sam_grid_random.py".
Before running it, please edit the variables from the argparse. 
It also provides detailed descriptions of all the variables, so please consider reading these carefully.

Here we give an example: running with Co and REE commodities with pixel interval of point prompt 35, and the number of random point prompts 5, 10, 20, 30, and 50:
```
CUDA_VISIBLE_DEVICES=0 python mo_sam_grid.py --comm_list Co REE --iv_list 35 --rn_list 5 10 20 30 50
```

## Step-5: Checking Your Results.

All the results will be in the "./Results/" folder, including the IoU evaluation scores and the visualizations.

Codebase References:

https://github.com/facebookresearch/segment-anything.
