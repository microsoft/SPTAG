# Research references

[Documentation index](README.md)

Please cite the work relevant to the feature used, rather than treating every
integration as part of the original SPTAG release.

## SPIRE

[Scalable Distributed Vector Search via Accuracy Preserving Index Construction](https://arxiv.org/abs/2512.17264),
arXiv:2512.17264, submitted December 19, 2025.

SPIRE is research associated with this repository's distributed vector-search
and index-construction work. The paper studies partition granularity and
accuracy-preserving recursive construction of multi-level indexes. See the
[distributed research workflow](../evaluation/distributed/Readme.md) for the
repository's experimental entry point. Paper results are not a validation of
every configuration or a claim that this worktree reproduces the full evaluation.

```bibtex
@misc{xu2025scalable,
  title={Scalable Distributed Vector Search via Accuracy Preserving Index Construction},
  author={Xu, Yuming and Zhang, Qianxi and Chen, Qi and Lu, Baotong and Li, Menghao and Adams, Philip and Li, Mingqin and Li, Zengzhong and Liu, Jing and Li, Cheng and Yang, Fan},
  year={2025},
  eprint={2512.17264},
  archivePrefix={arXiv},
  primaryClass={cs.DC},
  url={https://arxiv.org/abs/2512.17264}
}
```

## SPFresh

[SPFresh: Incremental In-Place Update for Billion-Scale Vector Search](https://dl.acm.org/doi/10.1145/3600006.3613166),
SOSP 2023.

```bibtex
@inproceedings{xu2023spfresh,
  title={SPFresh: Incremental In-Place Update for Billion-Scale Vector Search},
  author={Xu, Yuming and Liang, Hengyu and Li, Jin and Xu, Shuotao and Chen, Qi and Zhang, Qianxi and Li, Cheng and Yang, Ziyue and Yang, Fan and Yang, Yuqing and others},
  booktitle={Proceedings of the 29th Symposium on Operating Systems Principles},
  pages={545--561},
  year={2023}
}
```

## VBASE

[VBASE: Unifying Online Vector Similarity Search and Relational Queries via Relaxed Monotonicity](https://www.usenix.org/system/files/osdi23-zhang-qianxi_1.pdf),
OSDI 2023.

```bibtex
@inproceedings{zhang2023vbase,
  title={VBASE: Unifying Online Vector Similarity Search and Relational Queries via Relaxed Monotonicity},
  author={Zhang, Qianxi and Xu, Shuotao and Chen, Qi and Sui, Guoxin and Xie, Jiadong and Cai, Zhizhen and Chen, Yaoqi and He, Yinxuan and Yang, Yuqing and Yang, Fan and others},
  booktitle={17th USENIX Symposium on Operating Systems Design and Implementation},
  year={2023}
}
```

## SPANN

```bibtex
@inproceedings{ChenW21,
  title={SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search},
  author={Chen, Qi and Zhao, Bing and Wang, Haidong and Li, Mingqin and Liu, Chuanjie and Li, Zengzhong and Yang, Mao and Wang, Jingdong},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

## Original SPTAG

```bibtex
@manual{ChenW18,
  title={SPTAG: A library for fast approximate nearest neighbor search},
  author={Chen, Qi and Wang, Haidong and Li, Mingqin and Ren, Gang and Li, Scarlett and Zhu, Jeffery and Li, Jason and Zhang, Lintao and Yang, Mao and Wang, Jingdong},
  url={https://github.com/microsoft/SPTAG},
  year={2018}
}
```

## Earlier graph-search work

- Jingdong Wang and Shipeng Li. *Query-driven iterated neighborhood graph search
  for large scale indexing*. ACM Multimedia, 2012, pages 179-188.
- Jing Wang, Jingdong Wang, Gang Zeng, Zhuowen Tu, Rui Gan, and Shipeng Li.
  *Scalable k-NN graph construction for visual descriptors*. CVPR, 2012,
  pages 1106-1113.
- Jingdong Wang, Naiyan Wang, You Jia, Jian Li, Gang Zeng, Hongbin Zha, and
  Xian-Sheng Hua. *Trinary-Projection Trees for Approximate Nearest Neighbor
  Search*. IEEE TPAMI 36(2), 2014, pages 388-403.
