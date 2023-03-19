import logging
import time
from argparse import ArgumentParser
from datetime import datetime
from os.path import exists
import os
import torch
import numpy as np
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
import pyro.contrib.examples.polyphonic_data_loader as poly
import torch.nn as nn
import config
import pickle
from models import DMM


class Train():
    def __init__(self, model, train_package, args, log_path):
        self.dmm = model
        self.args = args
        self.log_path = log_path
        self.train_package = train_package

    def setup_logging(self):
        if not os.path.exists('./Log'):
            os.makedirs('./Log')
        logging.basicConfig(level=logging.DEBUG, format="%(message)s", filename=self.log_path, filemode="w")
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger("").addHandler(console)
        logging.info('Train_set:{}\nLearning Rate:{}\nBatch Size:{}\nEpochs:{}\n'.format(
            self.args.dataset, self.args.lr, self.args.bs, self.args.epochs
        ))

    def setup_adam(self):
        params = {
            "lr": self.args.lr,
            "betas": (0.96, 0.999),
            "clip_norm": 10,
            "lrd": self.args.lr_decay,
            "weight_decay": self.args.weight_decay,
        }
        self.adam = ClippedAdam(params)

    def setup_inference(self):
        elbo = Trace_ELBO()
        self.svi = SVI(self.dmm.model, self.dmm.guide, self.adam, loss=elbo)

    def save_checkpoint(self, name):
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)
        torch.save(self.dmm.state_dict(), self.args.save_root + name)

    def load_checkpoint(self):
        assert exists(self.args.load_opt) and exists(self.args.load_model), "Load model: path error"
        logging.info("Loading model from %s" % self.args.load_model)
        self.dmm.load_state_dict(torch.load(self.args.load_model))
        self.adam.load(self.args.load_opt)

    def prepare_train_data(self):
        """ Organize data for training loop """
        _train = self.train_package['train']
        _info = self.train_package['info']['train']
        dic = {'sequences': None, 'sequence_lengths': None, 'images': None}
        scanpath_length = _info['scanpath_length']
        num_scanpath = _info['num_scanpath']
        image_index = np.zeros((num_scanpath))

        scanpath_set = np.zeros([num_scanpath, scanpath_length, 3])
        length_set = (np.ones(num_scanpath) * _info['scanpath_length']).astype(int)
        image_set = torch.zeros([num_scanpath, 3, config.image_size[0], config.image_size[1]])

        index, img_index = 0, 0
        for instance in _train:
            scanpaths = _train[instance]['scanpaths']
            for j in range(scanpaths.shape[0]):
                scanpath_set[index] = scanpaths[j]
                image_index[index] = img_index
                image_set[index] = _train[instance]['image']
                index += 1
            img_index += 1

        dic['sequences'] = torch.from_numpy(scanpath_set).float()
        dic['sequence_lengths'] = torch.from_numpy(length_set.astype(int))
        dic['image_index'] = torch.from_numpy(image_index.astype(int))
        dic['images'] = image_set

        return dic

    def get_mini_batch(self, mini_batch_indices, sequences, seq_lengths, images, cuda=False):
        """ Get mini batch for training """

        # get the sequence lengths of the mini-batch
        seq_lengths = seq_lengths[mini_batch_indices]
        # sort the sequence lengths: from max to min
        _, sorted_seq_length_indices = torch.sort(seq_lengths)
        sorted_seq_length_indices = sorted_seq_length_indices.flip(0)
        sorted_seq_lengths = seq_lengths[sorted_seq_length_indices]
        sorted_mini_batch_indices = mini_batch_indices[sorted_seq_length_indices]

        T_max = torch.max(seq_lengths)
        mini_batch = sequences[sorted_mini_batch_indices, 0:T_max, :]
        mini_batch_images = images[sorted_mini_batch_indices]
        # this is the sorted mini-batch in reverse temporal order
        mini_batch_reversed = poly.reverse_sequences(mini_batch, sorted_seq_lengths)
        # get mask for mini-batch
        mini_batch_mask = poly.get_mini_batch_mask(mini_batch, sorted_seq_lengths)

        # cuda() here because need to cuda() before packing
        if cuda:
            mini_batch = mini_batch.cuda()
            mini_batch_mask = mini_batch_mask.cuda()
            mini_batch_reversed = mini_batch_reversed.cuda()
            mini_batch_images = mini_batch_images.cuda()

        # do sequence packing
        mini_batch_reversed = nn.utils.rnn.pack_padded_sequence(mini_batch_reversed, sorted_seq_lengths,
                                                                batch_first=True)

        return mini_batch, mini_batch_reversed, mini_batch_mask, sorted_seq_lengths, mini_batch_images

    def process_minibatch(self, epoch, which_mini_batch, shuffled_indices):
        # prepare a mini-batch and take a gradient step to minimize -elbo

        if self.args.annealing_epochs > 0 and epoch < self.args.annealing_epochs:
            # compute the KL annealing factor approriate for the current mini-batch in the current epoch
            min_af = self.args.minimum_annealing_factor
            annealing_factor = min_af + (1.0 - min_af) * (
                    float(which_mini_batch + epoch * self.N_mini_batches + 1)
                    / float(self.args.annealing_epochs * self.N_mini_batches)
            )
        else:
            # by default the KL annealing factor is unity
            annealing_factor = 1.0

        # compute which sequences in the training set we should grab
        mini_batch_start = which_mini_batch * self.args.bs
        mini_batch_end = np.min([(which_mini_batch + 1) * self.args.bs, self.N_sequences])
        mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]

        # grab a fully prepped mini-batch using the helper function in the data loader
        (mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, mini_batch_images) = \
            self.get_mini_batch(mini_batch_indices, self.sequences, self.seq_lengths, self.images, cuda=config.use_cuda)

        # do an actual gradient step
        loss = self.svi.step(
            scanpaths=mini_batch,
            scanpaths_reversed=mini_batch_reversed,
            mask=mini_batch_mask,
            scanpath_lengths=mini_batch_seq_lengths,
            images=mini_batch_images,
            annealing_factor=annealing_factor,
        )
        return loss

    def run(self):
        self.setup_adam()
        self.setup_inference()
        self.setup_logging()

        if self.args.load_opt is not None and self.args.load_model is not None:
            self.load_checkpoint()

        train_data = self.prepare_train_data()
        self.sequences = train_data["sequences"]
        self.seq_lengths = train_data["sequence_lengths"]
        self.images = train_data["images"]
        self.N_sequences = len(self.seq_lengths)
        self.N_time_slices = float(torch.sum(self.seq_lengths))
        self.N_mini_batches = int(self.N_sequences / self.args.bs +
                                  int(self.N_sequences % self.args.bs > 0))

        logging.info("N_train_data: %d\t avg. training seq. length: %.2f\t N_mini_batches: %d"
                     % (self.N_sequences, self.seq_lengths.float().mean(), self.N_mini_batches))

        times = [time.time()]

        for epoch in range(self.args.epochs):
            epoch_nll = 0.0
            shuffled_indices = torch.randperm(self.N_sequences)

            for which_mini_batch in range(self.N_mini_batches):
                epoch_nll += self.process_minibatch(epoch, which_mini_batch, shuffled_indices)

            # report training diagnostics
            times.append(time.time())
            epoch_time = times[-1] - times[-2]
            logging.info("[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)"
                         % (epoch, epoch_nll / self.N_time_slices, epoch_time))

            save_name = 'model_lr-{}_bs-{}_epoch-{}.pkl'.format(
                self.args.lr, self.args.bs, epoch)

            self.save_checkpoint(save_name)


if __name__ == '__main__':
    parser = ArgumentParser(description='ScanDMM')
    parser.add_argument('--seed', default=config.seed, type=int,
                        help='seed, default = 1234')
    parser.add_argument('--dataset', default='./Datasets/Sitzmann.pkl', type=str,
                        help='dataset path, default = ./Datasets/Sitzmann.pkl')
    parser.add_argument('--lr', default=config.learning_rate, type=float,
                        help='learning rate, default = 0.0003')
    parser.add_argument('--bs', default=config.mini_batch_size, type=int,
                        help='mini batch size, default = 64')
    parser.add_argument('--lr_decay', default=config.lr_decay, type=float,
                        help='learning rate decay, default = 0.99998')
    parser.add_argument('--epochs', default=config.num_epochs, type=int,
                        help='num_epochs, default = 500')
    parser.add_argument('--weight_decay', default=config.weight_decay, type=float,
                        help='learning rate decay, default = 2.0')
    parser.add_argument('--annealing_epochs', default=config.annealing_epochs, type=int,
                        help='KL annealing, default = 10')
    parser.add_argument('--minimum_annealing_factor', default=config.minimum_annealing_factor, type=float,
                        help='minimum KL annealing factor, default = 0.2')
    parser.add_argument('--load_model', default=None, type=str,
                        help='path of pre-trained model, default = None')
    parser.add_argument('--load_opt', default=None, type=str,
                        help='path of optimizer state, default = None')
    parser.add_argument('--save_root', default=config.save_root, type=str,
                        help='model save path, default = ./model/')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dmm = DMM(use_cuda=config.use_cuda)

    train_log = './Log/lr-{}_bs-{}_dy-{}_epo-{}_{}.txt'.format(
        args.lr, args.bs,
        args.lr_decay, args.epochs,
        datetime.now().strftime("%I:%M%p on %B %d, %Y"))

    train_dict = pickle.load(open(args.dataset, 'rb'))

    trainer = Train(dmm, train_dict, args, train_log)

    trainer.run()
