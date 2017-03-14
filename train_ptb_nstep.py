#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import division
from __future__ import print_function
import argparse
import copy

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import reporter as reporter_module


# Definition of a recurrent net for language modeling
class RNNForLM(chainer.Chain):

    def __init__(self, n_vocab, n_units, train=True):
        n_layer = 2
        super(RNNForLM, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l1=L.NStepLSTM(n_layer, n_units, n_units, 0.5, True),
            l2=L.Linear(n_units, n_vocab),
        )
        self.train = train
        self.n_layer = n_layer
        self.n_units = n_units

    def __call__(self, xs):
        x_len = [len(x) for x in xs]
        x_section = np.cumsum(x_len[:-1])
        ex = self.embed(F.concat(xs, axis=0))
        exs = F.split_axis(ex, x_section, 0, force_tuple=True)

        xp = self.xp
        volatile = xs[0].volatile
        hx = chainer.Variable(xp.zeros((self.n_layer, len(xs), self.n_units), dtype=xp.float32), volatile=volatile)
        cx = chainer.Variable(xp.zeros((self.n_layer, len(xs), self.n_units), dtype=xp.float32), volatile=volatile)
        _, _, ys = self.l1(hx, cx, exs, train=self.train)
        y = [self.l2(F.dropout(i, train=self.train)) for i in ys]
        return y


# Dataset iterator to create a batch of sequences at different positions.
# This iterator returns a pair of current words and the next words. Each
# example is a part of sequences starting from the different offsets
# equally spaced within the whole sequence.
class ParallelSequentialIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, bprop_len, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        self.bprop_len = bprop_len
        # Number of completed sweeps over the dataset. In this case, it is
        # incremented if every word is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        # Offsets maintain the position of each sequence in the mini-batch.
        self.offsets = [i * length // batch_size for i in range(batch_size)]
        # NOTE: this is not a count of parameter updates. It is just a count of
        # calls of ``__next__``.
        self.iteration = 0

    def __next__(self):
        # This iterator returns a list representing a mini-batch. Each item
        # indicates a different position in the original sequence. Each item is
        # represented by a pair of two word IDs. The first word is at the
        # "current" position, while the second word at the next position.
        # At each iteration, the iteration count is incremented, which pushes
        # forward the "current" position.
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            # If not self.repeat, this iterator stops at the end of the first
            # epoch (i.e., when all words are visited once).
            raise StopIteration
        cur_words = self.get_words()
        self.iteration += 1
        next_words = self.get_words()

        epoch = self.iteration * self.batch_size * self.bprop_len // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(cur_words, next_words))

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration * self.batch_size * self.bprop_len / len(self.dataset)

    def get_words(self):
        items = []
        for offset in self.offsets:
            start = (offset + self.iteration) % len(self.dataset)
            item = self.dataset[start : start+self.bprop_len]
            if start+self.bprop_len > len(self.dataset):
                items.append(np.concatenate((item, self.dataset[:start + self.bprop_len - len(self.dataset)])))
            else:
                items.append(item)
        return items

    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)


def convert(batch, device):
    if device is None:
        def to_device(x):
            return x
    elif device < 0:
        to_device = chainer.cuda.to_cpu
    else:
        def to_device(x):
            return chainer.cuda.to_gpu(x, device, chainer.cuda.Stream.null)

    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [to_device(x) for x in batch]
        else:
            xp = chainer.cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x) for x in batch[:-1]], dtype='i')
            concat_dev = to_device(concat)
            batch_dev = chainer.cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return tuple([to_device_batch([x for x, _ in batch]), to_device_batch([y for _, y in batch])])


# Custom updater for truncated BackProp Through Time (BPTT)
class BPTTUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, converter=convert, device=device)

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Progress the dataset iterator for bprop_len words at each iteration.
        # Get the next batch (a list of tuples of two word IDs)
        batch = train_iter.__next__()

        # Concatenate the word IDs to matrices and send them to the device
        # self.converter does this job
        # (it is chainer.dataset.concat_examples by default)
        xs, ts = self.converter(batch, self.device)

        # Compute the loss at this time step and accumulate it
        loss = optimizer.target([chainer.Variable(x) for x in xs], [chainer.Variable(t) for t in ts])

        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters


class BPTTEvaluator(training.extensions.Evaluator):

    def __init__(self, iterator, target, device):
        super(BPTTEvaluator, self).__init__(
            iterator, target, converter=convert, device=device)

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func or target

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                xs, ts = self.converter(batch, self.device)
                eval_func([chainer.Variable(x, volatile='on') for x in xs], [chainer.Variable(t, volatile='on') for t in ts])

            summary.add(observation)

        return summary.compute_mean()


def sum_softmax_cross_entropy(ys, ts):
    loss = 0
    for y, t in zip(ys, ts):
        loss += chainer.functions.softmax_cross_entropy(y, t)
    return loss


# Routine to rewrite the result dictionary of LogReport to add perplexity
# values
def compute_perplexity(result):
    result['perplexity'] = np.exp(result['main/loss'])
    if 'validation/main/loss' in result:
        result['val_perplexity'] = np.exp(result['validation/main/loss'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=35,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=39,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of LSTM units in each layer')
    args = parser.parse_args()

    # Load the Penn Tree Bank long word sequence dataset
    train, val, test = chainer.datasets.get_ptb_words()
    n_vocab = max(train) + 1  # train is just an array of integers
    print('#vocab =', n_vocab)

    if args.test:
        train = train[:100]
        val = val[:100]
        test = test[:100]

    train_iter = ParallelSequentialIterator(train, args.batchsize, args.bproplen)
    val_iter = ParallelSequentialIterator(val, 1, args.bproplen, repeat=False)
    test_iter = ParallelSequentialIterator(test, 1, args.bproplen, repeat=False)

    # Prepare an RNNLM model
    rnn = RNNForLM(n_vocab, args.unit)
    model = L.Classifier(rnn, lossfun=sum_softmax_cross_entropy)
    model.compute_accuracy = False  # we only want the perplexity
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        model.to_gpu()

    # Set up an optimizer
    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    # Set up a trainer
    updater = BPTTUpdater(train_iter, optimizer, args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    eval_model = model.copy()  # Model with shared params and distinct states
    eval_rnn = eval_model.predictor
    eval_rnn.train = False
    trainer.extend(BPTTEvaluator(
        val_iter, eval_model, device=args.gpu))

    interval = 10 if args.test else 500
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity,
                                        trigger=(interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'perplexity', 'val_perplexity']
    ), trigger=(interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(
        update_interval=1 if args.test else 10))
    trainer.extend(extensions.snapshot())
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    # Evaluate the final model
    print('test')
    evaluator = BPTTEvaluator(test_iter, eval_model, device=args.gpu)
    result = evaluator()
    print('test perplexity:', np.exp(float(result['main/loss'])))


if __name__ == '__main__':
    main()
