# ViT-TAD
Our paper is available in [ViT-TAD](https://arxiv.org/abs/2312.01897) 
# News
[2024.5.4] The code for THUMOS14 and ActivityNet-1.3 is updated. <br>
[2023.2.27] Our ViT-TAD has been accepted by CVPR2024. The code will be updated soon. <br>
# Overview
![Pipeline](./figs/pipeline.png)

# Environment preparation

**1:  Create environment**

```
conda env create -f requirements/vittad.yml
```

**2:  Activate environment**

```
conda activate vittad
```

**3:  Install other dependencies**

``` 
pip install -v -e .
```

# Data preparation

**1:  Download videos**

For **THUMOS14**, please check [BasicTAD](https://github.com/MCG-NJU/BasicTAD) for downloading videos.

For **ActivityNet-1.3**, please check [TALLFormer](https://github.com/klauscc/TALLFormer) for downloading videos.

**2:  Prepare videos**

