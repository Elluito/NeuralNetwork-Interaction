import copy
import sys
import typing

import networkx as ntx
import numpy as np
import torch

eps = sys.float_info.epsilon


def ism(f: typing.Callable, dim: int, ub: int, lb: int, is_NN: bool = False, use_grad: bool = False) -> typing.Tuple[
    typing.Union[typing.Iterable,torch.TensorType], typing.Iterable,
    typing.Iterable, typing.Iterable]:
    if not is_NN:
        temp = (ub + lb) / 2
        # pdb.set_trace()

        f_archive = np.zeros((dim, dim)) * np.nan
        fhat_archive = np.zeros((dim, 1)) * np.nan
        delta1 = np.zeros((dim, dim)) * np.nan
        delta2 = np.zeros((dim, dim)) * np.nan
        lambda_matrix = np.zeros((dim, dim)) * np.nan

        p1 = np.array([lb] * dim)
        fp1 = f(p1)
        counter = 0
        prev = 0
        prog = 0

        for i in range(dim - 1):
            if not np.isnan(fhat_archive[i]):
                fp2 = fhat_archive[i]
            else:
                p2 = copy.deepcopy(p1)
                p2[i] = copy.deepcopy(temp)
                fp2 = f(p2)
                fhat_archive[i] = fp2

            for j in range(i + 1, dim):
                counter += 1
                prev = prog
                prog = np.floor(counter / (dim * (dim - 1)) * 2 * 100)
                if prog % 5 == 0 and prog != prev:
                    print("Progress: {}%".format(prog))
                if not np.isnan(fhat_archive[j]):
                    fp3 = fhat_archive[j]
                else:
                    p3 = copy.copy(p1)
                    p3[j] = temp
                    fp3 = f(p3)
                    fhat_archive[j] = fp3
                p4 = copy.deepcopy(p1)
                p4[i] = temp
                p4[j] = temp
                fp4 = f(p4)
                f_archive[i, j] = fp4
                f_archive[j, i] = fp4
                d1 = fp2 - fp1
                d2 = fp4 - fp3
                delta1[i, j] = d1
                delta2[i, j] = d2
                # pdb.set_trace()
                lambda_matrix[i, j] = abs(d1 - d2)
            # pdb.set_trace()
        return lambda_matrix, fhat_archive, f_archive, fp1
    else:
        if use_grad:
            temp = (ub + lb) / 2
            # pdb.set_trace()

            f_archive = torch.zeros((dim, dim)) * torch.nan
            fhat_archive = torch.zeros((dim, 1),device=torch.device("cuda")) * torch.nan
            delta1 = torch.zeros((dim, dim)) * torch.nan
            delta2 = torch.zeros((dim, dim)) * torch.nan
            lambda_matrix = torch.zeros((dim, dim)) * torch.nan

            p1 = torch.tensor([lb] * dim, dtype=torch.float32,device=torch.device("cuda"))

            fp1 = f(p1)
            counter = 0
            prev = 0
            prog = 0

            for i in range(dim - 1):
                if not torch.isnan(fhat_archive[i]):
                    fp2 = fhat_archive[i]
                else:
                    p2 = copy.deepcopy(p1)
                    p2[i] = copy.deepcopy(temp)
                    p2.to(torch.device("cuda"))
                    fp2 = f(p2)

                    fhat_archive[i] = fp2

                for j in range(i + 1, dim):
                    counter += 1
                    prev = prog
                    prog = torch.tensor([counter // (dim * (dim - 1)) * 2 * 100])

                    if prog % 5 == 0 and prog != prev:
                        print("Progress: {}%".format(prog))
                    if not torch.isnan(fhat_archive[j]):
                        fp3 = fhat_archive[j]
                    else:
                        p3 = copy.copy(p1)
                        p3[j] = temp
                        p3.to(torch.device("cuda"))
                        fp3 = f(p3)
                        fhat_archive[j] = fp3
                    p4 = copy.deepcopy(p1)
                    p4[i] = temp
                    p4[j] = temp
                    p4.to(torch.device("cuda"))
                    fp4 = f(p4)
                    f_archive[i, j] = fp4
                    f_archive[j, i] = fp4
                    d1 = fp2 - fp1
                    d2 = fp4 - fp3
                    delta1[i, j] = d1
                    delta2[i, j] = d2
                    # pdb.set_trace()
                    lambda_matrix[i, j] = abs(d1 - d2)
                # pdb.set_trace()
            return lambda_matrix, fhat_archive, f_archive, fp1
        else:

            with torch.no_grad():
                temp = (ub + lb) / 2
                f_archive = torch.zeros((dim, dim)) * torch.nan
                fhat_archive = torch.zeros((dim, 1),device=torch.device("cuda")) * torch.nan
                delta1 = torch.zeros((dim, dim)) * torch.nan
                delta2 = torch.zeros((dim, dim)) * torch.nan
                lambda_matrix = torch.zeros((dim, dim)) * torch.nan

                p1 = torch.tensor([lb] * dim, dtype=torch.float32,device=torch.device("cuda"))
                fp1 = f(p1)
                counter = 0
                prev = 0
                prog = 0

                for i in range(dim - 1):
                    if not torch.isnan(fhat_archive[i]):
                        fp2 = fhat_archive[i]
                    else:
                        p2 = copy.deepcopy(p1)
                        p2[i] = copy.deepcopy(temp)
                        p2.to(torch.device("cuda"))
                        fp2 = f(p2)
                        fhat_archive[i] = fp2

                    for j in range(i + 1, dim):
                        counter += 1
                        prev = prog
                        prog = torch.tensor([counter // (dim * (dim - 1)) * 2 * 100])

                        if prog % 5 == 0 and prog != prev:
                            print("Progress: {}%".format(prog))
                        if not torch.isnan(fhat_archive[j]):
                            fp3 = fhat_archive[j]
                        else:
                            p3 = copy.copy(p1)
                            p3[j] = temp
                            p3.to(torch.device("cuda"))
                            fp3 = f(p3)
                            fhat_archive[j] = fp3
                        p4 = copy.deepcopy(p1)
                        p4[i] = temp
                        p4[j] = temp
                        p4.to(torch.device("cuda"))
                        fp4 = f(p4)
                        f_archive[i, j] = fp4
                        f_archive[j, i] = fp4
                        d1 = fp2 - fp1
                        d2 = fp4 - fp3
                        delta1[i, j] = d1
                        delta2[i, j] = d2
                        # pdb.set_trace()
                        lambda_matrix[i, j] = abs(d1 - d2)
                    # pdb.set_trace()
                return lambda_matrix, fhat_archive, f_archive, fp1


def dsm(fhat_archive, lambda_matrix, f_archive, fp1, dim):
    F1 = np.ones((dim, dim)) * fp1
    F2 = np.tile(np.transpose(fhat_archive), (dim, 1))
    F3 = np.tile(fhat_archive, (1, dim))
    F4 = f_archive

    FS = np.stack((F1, F2, F3, F4), axis=2)
    F_max = np.max(FS, axis=2)
    F_min = np.min(FS, axis=2)
    FS = np.stack(((F1 + F4), (F2 + F3)), axis=2)
    F_max_inf = np.max(FS, axis=2)

    theta = np.ones((dim, dim)) * np.nan
    reliable_calcs = 0
    nuM = eps / 2

    gamma = lambda n: (n * nuM) / (1 - n * nuM)
    errlb = gamma(2) * F_max_inf
    errub = gamma(dim ** 0.5) * F_max

    # I did a little modification instead of lambda_matrix < errlb I use lambda_matrix <= errlb because otherwise the
    # THETA matrix outputs all 1
    I1 = lambda_matrix < errlb
    theta[I1] = 0
    I2 = lambda_matrix > errub
    theta[I2] = 1

    si1 = np.sum(I1)
    si3 = np.sum(I2)

    I0 = lambda_matrix == 0
    C0 = np.sum(I0)
    count_seps = np.sum(np.logical_and(np.logical_not(I0), I1))
    count_non_seps = np.sum(I2)
    reliable_calcs = count_seps + count_non_seps
    w1 = (count_seps + C0) / (C0 + reliable_calcs)
    w2 = (count_non_seps) / (C0 + reliable_calcs)
    epsilon = w1 * errlb + w2 * errub
    # did a little modification instead of lambda_matrix < errub I use lambda_matrix <= errub because otherwise the
    # THETA matrix outputs all 1
    grayind = np.logical_and(lambda_matrix < errub, lambda_matrix > errlb)
    grayindsum = np.sum(grayind)
    # FIXME: This line of code is giving problems since epsilon is a very small NEGATIVE number and obiously 0 is
    # bigger than is so this line outputs all 1s
    AdjTemp = lambda_matrix > epsilon
    idx = np.isnan(theta)
    theta[idx] = AdjTemp[idx]
    theta = np.logical_or(theta, np.transpose(theta))
    theta[np.eye(dim, dtype=np.bool)] = 1
    temp_graph = ntx.convert_matrix.from_numpy_matrix(theta)
    components = list(ntx.algorithms.components.connected_components(temp_graph))

    h = lambda x: len(x) == 1
    sizeone = np.array(list(map(h, components)))
    indices = np.array(range(len(components)))
    indices_sep = indices[sizeone]

    seps = []
    non_seps = []

    for elem in indices_sep:
        seps.append(components[elem])
    for elem in indices[np.logical_not(sizeone)]:
        non_seps.append(components[elem])

    return non_seps, seps, theta, epsilon


def read_theta_matrix_function(number_function=1):
    import scipy.io
    if number_function < 10:
        mat = scipy.io.loadmat('f0{}.mat'.format(number_function))
        return mat
    else:
        mat = scipy.io.loadmat('f{}.mat'.format(number_function))
        return mat


def rho_metrics(theta, real_theta):
    dim = theta.shape[0]
    r1 = (np.sum(theta * real_theta - np.eye(dim)) / np.sum(real_theta - np.eye(dim))) * 100
    # pdb.set_trace()
    r2 = np.sum((np.ones((dim, dim)) - theta) * (np.ones((dim, dim)) - real_theta)) / np.sum(
        np.ones((dim, dim)) - real_theta) * 100
    r3 = np.sum(np.triu(np.ones((dim, dim)) - np.abs(theta - real_theta)) - np.eye(dim)) * 2 / (dim * (dim - 1)) * 100

    return r1, r2, r3


if __name__ == '__main__':
    with open("f1", "r+b") as f:
        f1 = pickle.load(f)
    with open("info_f1", "r+b") as f:
        info = pickle.load(f)
    dim = info['dimension']
    sol = info['lower'] + np.random.rand(dim) * (info['upper'] - info['lower'])
    salida = f1(sol * info['lower'] * np.ones(dim))
    print("Salida: {}".format(salida))
    lambda_matrix, fhat_archive, f_archive, fp1 = ism(f1, dim, info['upper'], info['lower'])
    non_seps, seps, theta, epsilon = dsm(fhat_archive, lambda_matrix, f_archive, fp1, dim)
    len(non_seps)
