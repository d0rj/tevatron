# Tevatron
Tevatron is a simple and efficient toolkit for training and running dense retrievers with deep language models. 
The toolkit has a modularized design for easy research; a set of command line tools are also provided for fast
development and testing. A set of easy-to-use interfaces to Huggingface's state-of-the-art pre-trained transformers
ensures Tevatron's superior performance.

*Tevatron is currently under initial development stage. We will be actively adding new features and API changes
may happen. Suggestions, feature requests and PRs are welcomed.*

### Быстрый запуск

```bash
cd src
```

```bash
sh dpr.sh
```

Редактируйте параметры оттуда.

## Features
- Command line interface for dense retriever training/encoding and dense index search.
- Flexible and extendable Pytorch retriever models. 
- Highly efficient Trainer, a subclass of  Huggingface Trainer, that naively support training performance features like mixed precision and distributed data parallel.
- Fast and memory-efficient train/inference data access based on memory mapping with Apache Arrow through Huggingface datasets.
- Jax/Flax training/encoding on TPU

## Installation
First install neural network and similarity search backends, 
namely Pytorch (or Jax) and FAISS.
Check out the official installation guides for [Pytorch](https://pytorch.org/get-started/locally/#start-locally)
, [Jax](https://github.com/google/jax) / [Flax](https://flax.readthedocs.io/en/latest/installation.html) 
and [FAISS](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) accordingly.

Then install Tevatron with pip,
```bash
pip install tevatron
```

Or typically for development and research, clone this repo and install as editable,
```
git https://github.com/texttron/tevatron
cd tevatron
pip install --editable .
```

> Note: The current code base has been tested with, `torch==1.10.1`, `faiss-cpu==1.7.2`, `transformers==4.15.0`, `datasets==1.17.0`

Optionally, you can also install GradCache to support our gradient cache feature during training by:
```bash
git clone https://github.com/luyug/GradCache
cd GradCache
pip install .
```

## Documentation
- [**Please view the documentation here**](http://tevatron.ai/)


## Examples
In the `/examples` folder, we provided full pipeline instructions for various IR/QA tasks.

## Citation
If you find Tevatron helpful, please consider citing our [paper](https://arxiv.org/abs/2203.05765).
```
@article{Gao2022TevatronAE,
  title={Tevatron: An Efficient and Flexible Toolkit for Dense Retrieval},
  author={Luyu Gao and Xueguang Ma and Jimmy J. Lin and Jamie Callan},
  journal={ArXiv},
  year={2022},
  volume={abs/2203.05765}
}
```

## Contacts
If you have a toolkit specific question, feel free to open an issue. 

You can also reach out to us for general comments/suggestions/questions through email.
- Luyu Gao luyug@cs.cmu.edu
- Xueguang Ma x93ma@uwaterloo.ca
