import os
import time
import glob

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from torchtext import data
from torchtext import datasets

import os
from argparse import ArgumentParser

from models import *

criterion = nn.CrossEntropyLoss()

def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext IMDB DAN example')
    parser.add_argument('path', type=str)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_embed', type=int, default=100)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--dev_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--dp_ratio', type=int, default=0.2)
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--word_vectors', type=str, default='glove.6B.100d')
    args = parser.parse_args()
    return args

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inputs = data.Field(lower=True, tokenize='spacy')
    answers = data.Field(sequential=False)

    train_path = os.path.join(args.path, 'train')
    test_path = os.path.join(args.path, 'test')
    valid_path = os.path.join(args.path, 'val')

    train = datasets.IMDB(train_path, inputs, answers)
    test = datasets.IMDB(test_path, inputs, answers)
    valid = datasets.IMDB(valid_path, inputs, answers)

    inputs.build_vocab(train, valid, test)
    inputs.vocab.load_vectors(args.word_vectors)

    answers.build_vocab(train)

    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, valid, test), batch_size=args.batch_size, device=device)

    n_embed = len(inputs.vocab)
    d_out = len(answers.vocab)

    model = DAN(n_embed=n_embed , d_embed=args.d_embed, d_hidden=256, d_out=d_out, dp=0.2, embed_weight=inputs.vocab.vectors)
    model.to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)

    acc, val_loss = evaluate(dev_iter, model)
    best_acc = acc

    print(
        'epoch |   %        |  loss  |  avg   |val loss|   acc   |  best  | time | save |')
    print(
        'val   |            |        |        | {:.4f} | {:.4f} | {:.4f} |      |      |'.format(
            val_loss, acc, best_acc))

    iterations = 0
    last_val_iter = 0
    train_loss = 0
    start = time.time()
    for epoch in range(args.epochs):
        train_iter.init_epoch()
        n_correct, n_total, train_loss = 0, 0, 0
        last_val_iter = 0
        for batch_idx, batch in enumerate(train_iter):
            # switch model to training mode, clear gradient accumulators
            model.train();
            opt.zero_grad()

            iterations += 1

            # forward pass
            answer = model(batch)
            loss = criterion(answer, batch.label)

            loss.backward();
            opt.step()

            train_loss += loss.item()
            print('\r {:4d} | {:4d}/{} | {:.4f} | {:.4f} |'.format(
                epoch,  args.batch_size * (batch_idx + 1), len(train), loss.item(),
                                             train_loss / (iterations - last_val_iter)), end='')

            if iterations > 0 and iterations % args.dev_every == 0:
                acc, val_loss = evaluate(dev_iter, model)

                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), args.save_path)
                    _save_ckp = '*'

                print(
                    ' {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:4s} |'.format(
                        val_loss, acc, best_acc, (time.time() - start) / 60,
                        _save_ckp))

                train_loss = 0
                last_val_iter = iterations

def evaluate(loader, model):
    model.eval()
    loader.init_epoch()

    # calculate accuracy on validation set
    n_correct, n = 0, 0
    losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            answer = model(batch)
            n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
            n += answer.shape[0]
            loss = criterion(answer, batch.label)
            losses.append(loss.data.cpu().numpy())
    acc = 100. * n_correct / n
    loss = np.mean(losses)

    return acc, loss

if __name__ == '__main__':
    args = get_args()
    train(args)
