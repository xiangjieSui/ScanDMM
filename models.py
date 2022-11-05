from sphere_cnn import Sphere_CNN
import torch.nn as nn
import torch
import pyro
import pyro.contrib.examples.polyphonic_data_loader as poly
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import *


class Emitter(nn.Module):
    """ p(x_t | z_t) """

    def __init__(self, gaze_dim, z_dim, hidden_dim):
        super().__init__()
        self.lin_em_z_to_hidden = nn.Linear(z_dim, hidden_dim)
        self.lin_hidden_to_gaze = nn.Linear(hidden_dim, gaze_dim)
        self.lin_gaze_sig = nn.Linear(gaze_dim, gaze_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.sigmod = nn.Sigmoid()

    def forward(self, z_t):
        """ We normalize x_mu to [-1,1] using sigmoid()*2 -1 """
        mu = self.sigmod(self.lin_hidden_to_gaze(self.relu(self.lin_em_z_to_hidden(z_t)))) * 2 - 1
        sigma = self.softplus(self.lin_gaze_sig(self.relu(mu)))
        return mu, sigma


class GatedTransition(nn.Module):
    """ p(z_t | z_{t-1}) """

    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.lin_gate_z_to_hidden_dim = nn.Linear(z_dim, hidden_dim)
        self.lin_gate_hidden_dim_to_z = nn.Linear(hidden_dim, z_dim)
        self.lin_trans_2z_to_hidden = nn.Linear(2 * z_dim, hidden_dim)
        self.lin_trans_hidden_to_z = nn.Linear(hidden_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_mu = nn.Linear(z_dim, z_dim)
        self.lin_z_to_mu.weight.data = torch.eye(z_dim)
        self.lin_z_to_mu.bias.data = torch.zeros(z_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, img_feature=None):
        """ Compute _z_t """
        z_t_1_img = torch.cat((z_t_1, img_feature), dim=1)
        _z_t = self.lin_trans_hidden_to_z(self.relu(self.lin_trans_2z_to_hidden(z_t_1_img)))

        ' Uncertainty weighting '
        weight = torch.sigmoid(self.lin_gate_hidden_dim_to_z(self.relu(self.lin_gate_z_to_hidden_dim(z_t_1))))
        ' Gaussian parameters '
        mu = (1 - weight) * self.lin_z_to_mu(z_t_1) + weight * _z_t
        sigma = self.softplus(self.lin_sig(self.relu(_z_t)))
        return mu, sigma


class Combiner(nn.Module):
    """ q(z_t | z_{t-1}, x_{t:T}) """

    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        self.lin_comb_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_mu = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_sig = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        h_combined = 0.5 * (self.tanh(self.lin_comb_z_to_hidden(z_t_1)) + h_rnn)
        mu = self.lin_hidden_to_mu(h_combined)
        sigma = self.softplus(self.lin_hidden_to_sig(h_combined))
        return mu, sigma


class DMM(nn.Module):

    def __init__(
            self,
            input_dim=3,
            z_dim=100,
            emission_dim=100,
            transition_dim=200,
            rnn_dim=600,
            num_layers=1,
            rnn_dropout_rate=0.1,
            use_cuda=False,
    ):
        super().__init__()
        self.cnn = Sphere_CNN(out_put_dim=z_dim)
        self.emitter = Emitter(input_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, transition_dim)
        self.combiner = Combiner(z_dim, rnn_dim)
        self.input_to_z_dim = nn.Linear(input_dim, z_dim)
        self.twoZ_to_z_dim = nn.Linear(2 * z_dim, z_dim)
        self.tanh = nn.Tanh()

        rnn_dropout_rate = 0.0 if num_layers == 1 else rnn_dropout_rate
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=rnn_dim,
            nonlinearity="relu",
            batch_first=True,
            bidirectional=False,
            num_layers=num_layers,
            dropout=rnn_dropout_rate,
        )

        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    # the model p(x_{1:T} | z_{1:T}), p(z_{1:T})
    def model(self, scanpaths, scanpaths_reversed, mask, scanpath_lengths, images=None, annealing_factor=1.0, predict=False):
        """ We use the mask() method to deal with variable-length scanapaths
            (i.e. different sequences have different lengths) """

        T_max = scanpaths.size(1)
        pyro.module("dmm", self)

        # state initialization
        z_prev = self.z_0.expand(scanpaths.size(0), self.z_0.size(0))
        z_prev = self.tanh(self.twoZ_to_z_dim(
            torch.cat((z_prev, self.tanh(self.input_to_z_dim(scanpaths[:, 0, :]))), dim=1)
        ))

        img_features = self.cnn(images)

        with pyro.plate("z_minibatch", len(scanpaths)):
            for t in pyro.markov(range(1, T_max + 1)):
                # Gaussian parameters of z
                z_mu, z_sigma = self.trans(z_prev, img_features)

                # Sample z_t according to N(z_loc, z_scale)
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample("z_%d" % t, dist.Normal(z_mu, z_sigma)
                                      .mask(mask[:, t - 1: t]).to_event(1))

                # Gaussian parameters of x
                x_mu, x_sigma = self.emitter(z_t)

                # Training
                if not predict:
                    pyro.sample("obs_x_%d" % t, dist.Normal(x_mu, x_sigma)
                                .mask(mask[:, t - 1: t]).to_event(1), obs=scanpaths[:, t - 1, :])
                # Test, removing obs
                else:
                    pyro.sample("obs_x_%d" % t, dist.Normal(x_mu, x_sigma)
                                .mask(mask[:, t - 1: t]).to_event(1))
                z_prev = z_t

    # the guide q(z_{1:T} | x_{1:T}) (i.e. the variational distribution)
    def guide(self, scanpaths, scanpaths_reversed, mask, scanpath_lengths, images=None, annealing_factor=1.0):

        T_max = scanpaths.size(1)
        pyro.module("dmm", self)

        h_0_contig = self.h_0.expand(1, scanpaths.size(0), self.rnn.hidden_size).contiguous()

        rnn_output, _ = self.rnn(scanpaths_reversed, h_0_contig)
        rnn_output = poly.pad_and_reverse(rnn_output, scanpath_lengths)

        z_prev = self.z_0.expand(scanpaths.size(0), self.z_0.size(0))
        z_prev = self.tanh(self.twoZ_to_z_dim(
            torch.cat((z_prev, self.tanh(self.input_to_z_dim(scanpaths[:, 0, :]))), dim=1)
        ))

        with pyro.plate("z_minibatch", len(scanpaths)):
            for t in pyro.markov(range(1, T_max + 1)):

                # assemble the distribution q(z_t | z_{t-1}, x_{t:T})

                z_mu, z_sigma = self.combiner(z_prev, rnn_output[:, t - 1, :])

                z_dist = dist.Normal(z_mu, z_sigma)
                assert z_dist.event_shape == ()
                assert z_dist.batch_shape[-2:] == (len(scanpaths), self.z_0.size(0))

                with pyro.poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(
                        "z_%d" % t,
                        z_dist.mask(mask[:, t - 1: t]).to_event(1),
                    )
                z_prev = z_t
