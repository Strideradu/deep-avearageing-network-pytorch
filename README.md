# Deep Averaging Networks (DAN) in PyTorch
pytorch implementation code for model described in
<http://cs.umd.edu/~miyyer/pubs/2015_acl_dan.pdf> along with IMDB dataset (<http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz>). 
feel free to email me at dunan@msu.edu with any comments/problems/questions/suggestions.

The original implementation is [Deep Averaging Networks (DAN)](https://github.com/miyyer/dan)

### dependencies: 
- python 3, pytorch 1.0, torchtext 0.4, spacy

### instructions

```angular2html
PyTorch/torchtext IMDB DAN example

positional arguments:
  path                  path to the IMDB dataset should have

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       epochs (default: 50)
  --batch_size BATCH_SIZE (default: 128)
  --d_embed D_EMBED     dimension of embeddings (default:100)
  --lr LR               (default: 0.001)
  --dev_every DEV_EVERY
  --dp_ratio DP_RATIO
  --save_path SAVE_PATH
                        path to save the model
  --word_vectors WORD_VECTORS
                        support the following choices: charngram.100d
                        fasttext.en.300d fasttext.simple.300d glove.42B.300d
                        glove.840B.300d glove.twitter.27B.25d
                        glove.twitter.27B.50d glove.twitter.27B.100d
                        glove.twitter.27B.200d glove.6B.50d glove.6B.100d
                        glove.6B.200d glove.6B.300d (default: glove.6B.100d)

```
