# CAFD
Class-Aware Frechet Distance (CAFD) for GANs in Tensorflow. Source code for "An Improved Evaluation Framework for Generative Adversarial Networks" \[[pdf](https://arxiv.org/pdf/1803.07474.pdf)\].

# Dependencies
* python >= 3.5.0
* Tensorflow >= 1.4.0

# Encoder
* mnist \[[GoogleDrive](https://drive.google.com/file/d/1KAfpbl08fTuoFaUM0Wr0fcbCMoBvQSW9/view)\]
* fashion-mnist \[[GoogleDrive](https://drive.google.com/file/d/16SdetBp35q7C4InWiOPY9yOeq_0iMV09/view)\]

# Usage
You should manually assign the path of the encoder in cafd.py.
To use CAFD, you can manually assign the two sources in cafd.py. 
```
python cafd.py
```
Also, you can directly use
```
python cafd.py folder1 folder2
```
or
```
python cafd.py feat1.csv feat2.csv
```
to compute the Class-Aware Frechet Distance (CAFD) between two distributions.

# Experiments
* The folder `celebA` contains the code for Fig. 3 in the paper.

* The folder `mnist` contains the code for Fig. 5 in the papar.

# Citation

If you find this code useful in your research, please cite:
```
@article{cafd2018,
  author       = {Shaohui Liu, Yi Wei, Jiwen Lu, Jie Zhou},
  title        = {An Improved Evaluation Framework for Generative Adversarial Networks},
  Journal      = {arXiv preprint arXiv:1803.07474},
  year         = {2018},
}
```
The first two authors share equal contributions.

