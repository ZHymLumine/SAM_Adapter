## SSM-SAM
Yiming Zhang, Tianang Leng, Kun Han, Xiaohui Xie

Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision![image](https://github.com/ZHymLumine/SSM-SAM/assets/47180638/5a07ce30-b690-4135-a6ce-4577378fa07a)
## Environment
This code was implemented with Python 3.8 and PyTorch 1.13.0. You can install all the requirements via:
```bash
pip install -r requirements.txt
```

## Quick Start
1. Download the pre-trained [SAM(Segment Anything)](https://github.com/facebookresearch/segment-anything) and put it in ./pretrained.
2. Training:
```bash
 python train.py --config configs/demo.yaml
```

4. Evaluation:
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]
```
