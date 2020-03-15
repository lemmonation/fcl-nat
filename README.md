# FCL-NAT

Code for our AAAI 2020 [paper](https://arxiv.org/pdf/1911.08717):

> Fine-Tuning by Curriculum Learning for Non-Autoregressive Neural Machine Translation      
> Junliang Guo, Xu Tan, Linli Xu, Tao Qin, Enhong Chen, Tie-Yan Liu

### Note

The code is mainly based on [tensor2tensor](https://github.com/tensorflow/tensor2tensor) and is tested under the following environments:

* tensorflow == 1.4
* tensor2tensor == 1.2.9

As the codebase of both tensorflow and tensor2tensor have changed drastically, we cannot guarantee the execution of our code on other versions. 
The core logic of our model is in `tensor2tensor/model/transformer_nat_cl_word.py`. We also provide a sample training script of our model in `scripts/train_nat_distill_wmt_ende_cl_word_set0.sh`.

The original version of this code is written by [Zhuohan Li](https://github.com/zhuohan123/hint-nart). We thank them a lot for sharing the code.

### Citation
```
@article{guo2019finetuning,
    title={Fine-Tuning by Curriculum Learning for Non-Autoregressive Neural Machine Translation},
    author={Guo, Junliang and Tan, Xu and Xu, Linli and Qin, Tao and Chen, Enhong and Liu, Tie-Yan},
    journal={arXiv preprint arXiv:1911.08717},
    year={2019}
}
```
