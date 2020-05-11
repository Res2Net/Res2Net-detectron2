# Res2Net for Panoptic Segmentation based on detectron2.

## Introduction

We propose a novel building block for CNNs, namely Res2Net, by constructing hierarchical residual-like connections within one single residual block. The Res2Net represents multi-scale features at a granular level and increases the range of receptive fields for each network layer.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">PQ</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: panoptic_fpn_R_50_1x -->
 <tr><td align="left"><a href="configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml">R50-FPN</a></td>
<td align="center">1x</td>
<td align="center">4.8</td>
<td align="center">37.6</td>
<td align="center">34.7</td>
<td align="center">39.4</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x/139514544/model_final_dbfeb4.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x/139514544/metrics.json">metrics</a></td>
</tr>
<!-- ROW: panoptic_fpn_R_50_3x -->
 <tr><td align="left"><a href="configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml">R50-FPN</a></td>
<td align="center">3x</td>
<td align="center">4.8</td>
<td align="center">40.0</td>
<td align="center">36.5</td>
<td align="center">41.5</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/metrics.json">metrics</a></td>
</tr>
<!-- ROW: panoptic_fpn_R_101_3x -->
 <tr><td align="left"><a href="configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml">R101-FPN</a></td>
<td align="center">3x</td>
<td align="center">6.0</td>
<td align="center">42.4</td>
<td align="center">38.5</td>
<td align="center">43.0</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/metrics.json">metrics</a></td>
</tr>
<!-- ROW: panoptic_fpn_R2_101_3x -->
 <tr><td align="left"><a href="configs/COCO-PanopticSegmentation/panoptic_fpn_R2_101_3x.yaml">Res2Net101-FPN</a></td>
<td align="center">3x</td>
<td align="center">6.0</td>
<td align="center">44.0</td>
<td align="center">39.6</td>
<td align="center">44.5</td>
<td align="center"><a href="https://mailnankaieducn-my.sharepoint.com/:u:/g/personal/shgao_mail_nankai_edu_cn/EU024RDiIxtJs2xz2zl_7bkBRXiPFcRukFLcB4gVYxzasw?e=PHU5jk">model</a>&nbsp;|&nbsp;<a href="results/panoptic_seg_res2net101_fpn_x3.txt">metrics</a></td>
</tr>
</tbody></table>

- Res2Net101 has the similar parameters and FLOPs compared with ResNet101.
- We only test and show results of Panoptic Segmentation on detectron2, the detection and instance segmentation on detectron2 is also supported in this repo.
- Res2Net ImageNet pretrained models are in [Res2Net-PretrainedModels](https://github.com/Res2Net/Res2Net-PretrainedModels).
- More applications of Res2Net are in [Res2Net-Github](https://github.com/Res2Net/).

## Useage
- Use the tools/convert-torchvision-to-d2.py to transfer the ImageNet pretrained model of Res2Net101 to detectron2 supported format. Or just download the converted model from
this [link](https://mailnankaieducn-my.sharepoint.com/:u:/g/personal/shgao_mail_nankai_edu_cn/EZhtWgMRlxpGtJmtJ2zP1_QBqvmu_FJ05vUgOq30ElT9yg?e=e5FRD8).
- Use the command to train:
```
./tools/train_net.py --num-gpus 8 --config-file configs/COCO-PanopticSegmentation/panoptic_fpn_R2_101_3x.yaml
```

## Citation
If you find this work or code is helpful in your research, please cite:
```
@article{gao2019res2net,
  title={Res2Net: A New Multi-scale Backbone Architecture},
  author={Gao, Shang-Hua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip},
  journal={IEEE TPAMI},
  year={2020},
  doi={10.1109/TPAMI.2019.2938758}, 
}
```

For more details of detectron2, please refer to
the detectron2 repo.

<img src=".github/Detectron2-Logo-Horz.svg" width="300" >

Detectron2 is Facebook AI Research's next generation software system
that implements state-of-the-art object detection algorithms.
It is a ground-up rewrite of the previous version,
[Detectron](https://github.com/facebookresearch/Detectron/),
and it originates from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).

<div align="center">
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>

### What's New
* It is powered by the [PyTorch](https://pytorch.org) deep learning framework.
* Includes more features such as panoptic segmentation, densepose, Cascade R-CNN, rotated bounding boxes, etc.
* Can be used as a library to support [different projects](projects/) on top of it.
  We'll open source more research projects in this way.
* It [trains much faster](https://detectron2.readthedocs.io/notes/benchmarks.html).

See our [blog post](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/)
to see more demos and learn about detectron2.

## Installation

See [INSTALL.md](INSTALL.md).

## Quick Start

See [GETTING_STARTED.md](GETTING_STARTED.md),
or the [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5).

Learn more at our [documentation](https://detectron2.readthedocs.org).
And see [projects/](projects/) for some projects that are built on top of detectron2.

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Detectron2 Model Zoo](MODEL_ZOO.md).


## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing Detectron2

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
