## Introduction
This project is design to count the number of person in an image using Resnet, FPN and Bayes Loss. 

## Acknowledgment

This project is based on BAYESIAN+ ([paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ma_Bayesian_Loss_for_Crowd_Count_Estimation_With_Point_Supervision_ICCV_2019_paper.pdf), [code](https://github.com/zhiheng-ma/Bayesian-Crowd-Counting)) and MMCCN ([paper](https://openaccess.thecvf.com/content/ACCV2020/papers/Peng_RGB-T_Crowd_Counting_from_Drone_A_Benchmark_and_MMCCN_Network_ACCV_2020_paper.pdf), [code](https://github.com/VisDrone/DroneRGBT)).

## Results

| Method | Resolution | MAE |
| VGG19 & BAYESIAN+    |   256×256    |  18.46 |
| MMCCN & BAYESIAN+    |   256×256    |  12.05 |
| MMCCN & FPN & BAYESIAN+    |   256×256    |  10.17 |


## Getting Started

### Installation

**Step 1: Environment Setup:**

```bash
conda create -n counter
conda activate counter
```

**Step 2: Install Dependencies:**
```bash
git clone https://github.com/Shuanglin-1126/counter.git
cd counter
pip install requirements.txt
```

counter recommends setting up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:

### Model Training and Inference

**Train:**
```bash
python train.py
```

**Inference:**
```bash
python test.py
```




