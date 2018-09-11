###############################################################################
## generate synthetic cause-effect pairs using neural networks 
## 
## Copyright (c) 2016 Facebook (David Lopez-Paz)
## 
## All rights reserved.
## 
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
## 
## 1. Redistributions of source code must retain the above copyright
##    notice, this list of conditions and the following disclaimer.
## 
## 2. Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimer in the
##    documentation and/or other materials provided with the distribution.
## 
## 3. Neither the names of NEC Laboratories American and IDIAP Research
##    Institute nor the names of its contributors may be used to endorse or
##    promote products derived from this software without specific prior
##    written permission.
## 
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
## ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
## LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
## CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
## SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
## INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
## CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
## ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
## POSSIBILITY OF SUCH DAMAGE.
## 
###############################################################################

import os
import torch
import argparse


distributions = {
    -1: "Unspecified",
    0: "Bernoulli",
    1: "Cauchy",
    2: "Exponential",
    3: "Geometric",
    4: "Log-Normal",
    5: "Normal",
    6: "Multinomial",
    7: "Uniform"
}


mechanism_types = {
    0: "Post-Additive",
    1: "Post-Multiplicative",
    2: "Pre-Additive",
    3: "Pre-Multiplicative",
    4: "Non-Linear"
}


def coin_flip(sides=2):
    return int(torch.zeros(1).random_(0, sides).item())


def write_ascii_matrix(matrix, fname):
    if matrix.dim() == 1:
        matrix = matrix.view(-1, 1)

    with open(fname, 'w') as f:
        for i in range(matrix.size(0)):
            for j in range(matrix.size(1)):
                f.write(str(float(matrix[i, j])) + " ")
            f.write("\n")
    return None


def normalize(x):
    return (x - x.mean()) / x.std()


def exogenous_variable(n, dist_type):
    x = torch.zeros(n, 1)

    if dist_type == 0:
        p = torch.zeros(1).uniform_(0, 1).item()
        x.bernoulli_(p)
    elif dist_type == 1:
        p = torch.zeros(1).uniform_(0.5, 2).item()
        x.cauchy_(0, p)
    elif dist_type == 2:
        p = torch.zeros(1).uniform_(-2, 2).item()
        x.exponential_(p)
    elif dist_type == 3:
        p = torch.zeros(1).uniform_(0, 1).item()
        x.geometric_(p)
    elif dist_type == 4:
        p = torch.zeros(1).uniform_(0.1, 2).item()
        x.log_normal_(0, p)
    elif dist_type == 5:
        p = 1
        x.normal_()
    elif dist_type == 6:
        p = int(torch.zeros(1).random_(2, 100).item())
        x.random_(0, p)
    elif dist_type == 7:
        p = 1
        x.uniform_()
    else:
        raise NotImplementedError

    return normalize(x).detach(), p


def function(dim, gamma, nh=1024):
    w1 = torch.randn(dim, nh).mul(gamma)
    b1 = torch.rand(1, nh).mul(2 * 3.1416)
    w2 = torch.randn(nh, 1)

    def handle(x):
        return torch.mm((torch.mm(x, w1) + b1).cos(), w2)

    return handle


def mechanism(mechanism_type, mechanism_smoothness=0.1):
    if mechanism_type != 4:
        f = function(1, mechanism_smoothness)
    else:
        f = function(2, mechanism_smoothness)

    if mechanism_type == 0:
        def handle(x, n):
            return f(x) + n
    elif mechanism_type == 1:
        def handle(x, n):
            return f(x) * n
    elif mechanism_type == 2:
        def handle(x, n):
            return f(x + n)
    elif mechanism_type == 3:
        def handle(x, n):
            return f(x * n)
    elif mechanism_type == 4:
        def handle(x, n):
            return f(torch.cat((x, n), 1))
    else:
        raise NotImplementedError

    return handle


def pair(n=1024, given_cause=None):
    if given_cause is None:
        cause_type = coin_flip(8)
        cause, p_cause = exogenous_variable(n, cause_type)
    else:
        cause_type = -1
        cause, p_cause = given_cause, 0

    noise_type = coin_flip(8)
    noise, p_noise = exogenous_variable(n, noise_type)

    mechanism_type = coin_flip(5)
    mecha = mechanism(mechanism_type)

    effect = normalize(mecha(cause, noise)).detach()

    if given_cause is None:
        if coin_flip(2):
            return pair(n, given_cause=effect)

    fmt_str = "{:>11}({:+.3f}) {:>11}({:+.3f}) {:>19}\n"
    description = fmt_str.format(distributions[cause_type],
                                 p_cause,
                                 distributions[noise_type],
                                 p_noise,
                                 mechanism_types[mechanism_type])

    return cause, effect, description


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Synthetic generator')
    parser.add_argument('--save_dir', type=str, default='data/')
    parser.add_argument('--n_pairs', type=int, default=500)
    parser.add_argument('--n_samples', type=int, default=1024)
    args = parser.parse_args()

    full_dir = os.path.join(args.save_dir, "pairs_synthetic")

    if not os.path.exists(full_dir):
        os.makedirs(full_dir)

    f_meta = open(os.path.join(full_dir, 'pairmeta.txt'), 'w')

    for n in range(args.n_pairs):
        fname_data = os.path.join(full_dir, 'pair{:04d}.txt'.format(n))
        fname_desc = os.path.join(full_dir, 'pair{:04d}.dsc'.format(n))

        cause, effect, pair_desc = pair(args.n_samples)
        write_ascii_matrix(torch.cat((cause, effect), 1), fname_data)

        with open(fname_desc, "w") as f:
            f.write(pair_desc)

        f_meta.write("{:04d} 1 1 2 2 1\n".format(n))

    f_meta.close()
