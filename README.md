## SSM-SAM
Yiming Zhang, Tianang Leng, Kun Han, Xiaohui Xie

Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision
## Environment
This code was implemented with Python 3.8 and PyTorch 1.13.0. You can install all the requirements via:
```bash
pip install -r requirements.txt
```

## Quick Start
1. Download the pre-trained [SAM(Segment Anything)](https://github.com/facebookresearch/segment-anything) and put it in ./pretrained.
2. Training:
```bash
 python train.py --config configs/sam-bit-b.yaml
```

4. Evaluation:
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]
```

## Citation

If you find our work useful in your research, please consider citing:

```
@misc{zhang2023selfsampling,
      title={Self-Sampling Meta SAM: Enhancing Few-shot Medical Image Segmentation with Meta-Learning}, 
      author={Yiming Zhang and Tianang Leng and Kun Han and Xiaohui Xie},
      year={2023},
      eprint={2308.16466},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
