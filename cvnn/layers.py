import math

import torch
import torch.nn as nn
from utilities import fix_dimensions

from data_utilities import get_distance


class phi_etadelta_kappa_mu_Layer(nn.Module):
    def __init__(self, Epsilon):
        super(phi_etadelta_kappa_mu_Layer, self).__init__()
        self.mu = mu_Layer(Epsilon)
        self.delta = delta_Layer(Epsilon)

    def forward(self, kappa, phi_0, pbhist, eta, it):
        mu0, mu1, mu0den, mu1den = self.mu(pbhist, phi_0, it)
        delta = self.delta(phi_0, it)
        return phi_0 + eta * delta * (kappa + (pbhist - mu0) ** 2 - (pbhist - mu1) ** 2)


class phi_etakappa_mu_Layer(nn.Module):
    def __init__(self, Epsilon):
        super(phi_etakappa_mu_Layer, self).__init__()
        self.mu = mu_Layer(Epsilon)

    def forward(self, kappa, phi_0, pbhist, eta, it):
        if phi_0.shape[1] == 1:
            mu0, mu1, mu0den, mu1den = self.mu(pbhist, phi_0, it)
            phi = phi_0 + eta * (
                kappa + torch.abs(pbhist - mu0) ** 2 - torch.abs(pbhist - mu1) ** 2
            )
        else:
            phi = torch.zeros_like(phi_0)
            for j in range(phi_0.shape[1]):
                mu0, mu1 = self.mu(pbhist, phi_0[:, j : j + 1, :, :], it)
                phi[:, j : j + 1, :, :] = phi_0[:, j : j + 1, :, :] + eta * (
                    kappa[:, j : j + 1, :, :] + (pbhist - mu0) - (pbhist - mu1)
                )
        return phi


class PhiEtaKappaMu3DLayer(nn.Module):
    def __init__(self, epsilon):
        super(PhiEtaKappaMu3DLayer, self).__init__()
        self.mu = mu_Layer(epsilon)

    def forward(self, kappa, phi_0, pbhist, eta, it):
        mu0, mu1, mu0den, mu1den = self.mu(pbhist, phi_0, it)
        phi = phi_0 + eta * (
            kappa + torch.abs(pbhist - mu0) ** 2 - torch.abs(pbhist - mu1) ** 2
        )
        return phi


class PhiEtaMu3DLayer(nn.Module):
    def __init__(self, epsilon):
        super(PhiEtaMu3DLayer, self).__init__()
        self.mu = mu_Layer(epsilon)

    def forward(self, phi_0, pbhist, eta, it):
        mu0, mu1, mu0den, mu1den = self.mu(pbhist, phi_0, it)
        phi_pt5 = phi_0 + eta * (
            torch.abs(pbhist - mu0) ** 2 - torch.abs(pbhist - mu1) ** 2
        )
        return phi_pt5


class phi_etakappa_Layer(nn.Module):
    def __init__(self):
        super(phi_etakappa_Layer, self).__init__()

    def forward(self, kappa, phi_0, eta):
        return phi_0 + eta * kappa


class etakappa_Layer(nn.Module):
    def __init__(self):
        super(etakappa_Layer, self).__init__()

    def forward(self, kappa, eta):
        return eta * kappa


class standardize_Layer(nn.Module):
    def __init__(self):
        super(standardize_Layer, self).__init__()

    def forward(self, z):
        bs = z.shape[0]
        z_min = z.view(bs, -1).min(1, keepdim=True)
        z_stnd = 0.8 * z.shape[3] * (z / z_min.values[:, :, None, None])
        return z_stnd


class mu_Layer(nn.Module):
    def __init__(self, Epsilon):
        super(mu_Layer, self).__init__()
        self.Heaviside = Heaviside_Layer(Epsilon)
        self.HeavisideDT = HeavisideDT_Layer(Epsilon)

    def forward(self, P, phi0, it):
        P = fix_dimensions(P)
        phi0 = fix_dimensions(phi0)
        H = self.Heaviside(phi0, it)
        HDT = self.Heaviside(phi0, it)

        if torch.sum(torch.isnan(H)) > 0:
            print("Heaviside is the issue i.e., gives NaN")
        mu1 = []
        mu0 = []
        mu1den = []
        mu0den = []
        for i in range(len(H)):
            H_i = H[i, 0]
            HDT_i = HDT[i, 0]
            P_i = P[i, 0]
            mu1den.append(torch.sum(torch.sum(H_i)))
            mu1.append(torch.sum(torch.sum(P_i * H_i)) / mu1den[i])

            mu0den.append(torch.sum(torch.sum(1 - HDT_i)))
            mu0.append(torch.sum(torch.sum(P_i * (1 - HDT_i))) / mu0den[i])

        bsize = len(mu0)
        if len(H.shape) == 4:
            sz = [bsize, 1, 1, 1]
        elif len(H.shape) == 5:
            sz = [bsize, 1, 1, 1, 1]

        mu0 = torch.FloatTensor(mu0)
        mu0 = mu0.reshape(sz)
        mu1 = torch.FloatTensor(mu1)
        mu1 = mu1.reshape(sz)
        return mu0, mu1, mu0den, mu1den


class mu_prime_Layer(nn.Module):
    def __init__(self, Epsilon):
        super(mu_prime_Layer, self).__init__()
        self.Heaviside = Heaviside_Layer(Epsilon)
        self.Heaviside_prime = Heaviside_prime_Layer(Epsilon)

    def forward(self, P, phi0, it, mu0, mu1, mu0den, mu1den):
        Hprime = self.Heaviside_prime(phi0, it)
        mu1prime = torch.zeros_like(Hprime)
        mu0prime = torch.zeros_like(Hprime)
        for i in range(len(Hprime)):
            Hprime_i = Hprime[i, 0, :, :]
            P_i = P[i, 0, :, :]
            mu1prime[i, 0, :, :] = (1 / mu1den[i]) * Hprime_i * (P_i - mu1[i])
            mu0prime[i, 0, :, :] = (1 / mu0den[i]) * Hprime_i * (mu0[i] - P_i)

        return mu0prime, mu1prime


class Heaviside_Layer(nn.Module):
    def __init__(self, Epsilon):
        super(Heaviside_Layer, self).__init__()
        self.Epsilon = Epsilon
        self.EpsilonP = 1

    def forward(self, z, it):
        Epsilon = self.Epsilon * (self.EpsilonP**it)
        H = 1 / 2 * (1 + z / Epsilon + 1 / math.pi * torch.sin(math.pi * z / Epsilon))
        H[z > Epsilon] = 1
        H[z < -Epsilon] = 0
        return H


class HeavisideDT_Layer(nn.Module):
    def __init__(self, Epsilon):
        super(HeavisideDT_Layer, self).__init__()
        self.Epsilon = Epsilon
        self.EpsilonP = 1

    def forward(self, z, it):
        Epsilon = self.Epsilon * (self.EpsilonP**it)
        H = 1 / 2 * (1 + z / Epsilon + 1 / math.pi * torch.sin(math.pi * z / Epsilon))
        H[z > Epsilon] = 1
        H[z < -Epsilon] = 0
        return H


class Heaviside_prime_Layer(nn.Module):
    def __init__(self, Epsilon):
        super(Heaviside_prime_Layer, self).__init__()
        self.Epsilon = Epsilon

    def forward(self, z, it):
        Epsilon = self.Epsilon
        H = 1 / (2 * Epsilon) * (1 + torch.cos(math.pi * z / Epsilon))
        H[z > Epsilon] = 0
        H[z < -Epsilon] = 0
        return H


class ReLUDCapped(nn.Module):
    def __init__(self, t0, t1):
        super(ReLUDCapped, self).__init__()
        self.t0 = t0
        self.t1 = t1

    def forward(self, z):
        z[z > self.t1] = self.t1
        z[z < self.t0] = self.t0
        return z


class delta_Layer(nn.Module):
    def __init__(self, Epsilon):
        super(delta_Layer, self).__init__()
        self.Epsilon = Epsilon
        self.EpsilonP = 1.2

    def forward(self, z, it):
        # z=z*(250/torch.abs(torch.min(z)))
        Epsilon = self.Epsilon * (self.EpsilonP**it)
        # H = (1 / (2*self.Epsilon)) * (1 + torch.cos(math.pi * z / self.Epsilon))
        # H[z > self.Epsilon] = 0
        # H[z < -self.Epsilon] = 0
        H = Epsilon / (math.pi * (z**2 + Epsilon**2))
        return H
