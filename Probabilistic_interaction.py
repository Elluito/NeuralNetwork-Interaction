# !pip
# install
# cec2013lsgo

import numpy as np
from scipy.stats import beta, bernoulli
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import pdb
from cec2013lsgo.cec2013 import Benchmark
import networkx as ntx
import numpy as np
import sys
import pdb
import copy
from copy import deepcopy

eps = sys.float_info.epsilon
permutation = np.array(
    [558, 633, 817, 554, 733, 637, 579, 49, 938, 20, 115, 373, 586, 826, 577, 693, 940, 325, 769, 880, 483, 352, 943,
     328, 776, 384,
     347, 143, 964, 920, 923, 797, 495, 364, 45, 624, 451, 753, 222, 859, 831, 621, 976, 493, 715, 12, 888, 632, 336,
     145, 46, 719, 468, 476, 2, 724, 413, 259, 378, 699, 198, 867, 739, 992, 829, 678, 126, 177, 372, 893, 187, 795,
     185, 744, 228, 315, 603, 113, 767, 777, 432, 673, 822, 457, 903, 703, 537, 959, 947, 453, 955, 157, 311, 911, 205,
     588, 518, 604, 423, 531, 836, 30, 16, 79, 511, 209, 711, 72, 341, 242, 346, 696, 551, 902, 939, 74, 694, 600, 290,
     791, 565, 233, 375, 567, 491, 602, 433, 141, 490, 743, 396, 270, 167, 389, 973, 350, 591, 252, 296, 500, 494, 989,
     519, 669, 506, 487, 568, 11, 718, 267, 645, 792, 741, 824, 289, 301, 677, 596, 251, 340, 95, 876, 672, 471, 41,
     124, 208, 339, 908, 975, 192, 246, 530, 265, 408, 957, 862, 330, 76, 48, 658, 535, 758, 264, 215, 592, 986, 905,
     100, 553, 19, 691, 770, 990, 995, 461, 702, 99, 613, 21, 935, 542, 105, 304, 207, 153, 92, 200, 825, 477, 333, 236,
     783, 180, 227, 573, 434, 623, 742, 561, 882, 539, 748, 550, 291, 142, 343, 193, 424, 735, 663, 161, 355, 811, 869,
     555, 931, 740, 310, 778, 309, 348, 85, 547, 523, 606, 913, 55, 941, 861, 763, 612, 501, 927, 651, 915, 80, 653,
     371, 206, 248, 698, 582, 78, 934, 94, 448, 482, 956, 590, 334, 469, 407, 107, 5, 268, 963, 338, 287, 849, 368, 462,
     314, 481, 221, 961, 260, 948, 636, 237, 472, 601, 196, 298, 148, 526, 398, 360, 981, 150, 801, 255, 154, 684, 50,
     611, 689, 852, 832, 695, 7, 646, 809, 106, 566, 415, 598, 823, 421, 962, 278, 281, 232, 879, 9, 668, 370, 480, 165,
     381, 102, 930, 950, 414, 418, 605, 865, 508, 34, 499, 737, 512, 466, 97, 926, 450, 122, 144, 164, 374, 578, 24,
     732, 687, 907, 422, 392, 802, 191, 397, 436, 1000, 112, 937, 690, 534, 887, 675, 269, 560, 319, 454, 773, 803, 53,
     203, 807, 705, 884, 323, 891, 297, 921, 430, 515, 985, 889, 589, 404, 616, 349, 969, 478, 59, 536, 321, 327, 210,
     854, 390, 219, 631, 833, 970, 111, 885, 58, 721, 35, 538, 752, 525, 183, 929, 685, 293, 682, 173, 110, 363, 890,
     644, 813, 960, 214, 498, 463, 147, 489, 139, 635, 886, 444, 155, 225, 473, 821, 312, 295, 108, 543, 63, 75, 686,
     608, 479, 280, 449, 266, 804, 900, 874, 738, 380, 235, 257, 201, 175, 521, 928, 664, 460, 557, 425, 137, 169, 912,
     458, 665, 760, 574, 784, 361, 25, 641, 790, 243, 10, 982, 445, 855, 442, 218, 828, 18, 572, 96, 475, 736, 549, 93,
     181, 282, 932, 427, 657, 130, 766, 66, 585, 277, 626, 583, 65, 27, 133, 514, 262, 67, 868, 459, 810, 814, 830, 533,
     840, 818, 101, 581, 279, 505, 316, 447, 294, 529, 443, 625, 638, 353, 486, 991, 850, 14, 507, 409, 968, 517, 31,
     43, 764, 395, 544, 464, 467, 492, 838, 391, 674, 400, 725, 785, 120, 376, 656, 402, 958, 659, 439, 204, 576, 977,
     68, 326, 419, 388, 465, 765, 666, 82, 630, 17, 132, 513, 798, 916, 455, 358, 135, 524, 881, 897, 746, 417, 942,
     121, 860, 303, 247, 313, 993, 787, 619, 231, 28, 226, 701, 706, 971, 618, 710, 546, 580, 629, 292, 793, 114, 168,
     806, 643, 342, 816, 273, 300, 84, 781, 966, 570, 756, 176, 749, 356, 847, 667, 318, 362, 949, 456, 878, 548, 224,
     166, 15, 470, 771, 774, 502, 726, 125, 4, 387, 620, 163, 757, 474, 275, 946, 335, 32, 564, 44, 299, 660, 952, 609,
     768, 254, 628, 750, 722, 197, 864, 676, 128, 617, 320, 683, 520, 936, 844, 584, 3, 895, 367, 917, 639, 174, 845,
     234, 195, 54, 974, 634, 984, 98, 812, 23, 171, 229, 587, 38, 717, 179, 679, 307, 189, 997, 720, 485, 337, 306, 516,
     552, 61, 484, 73, 671, 239, 158, 238, 230, 933, 440, 81, 136, 211, 283, 509, 799, 86, 199, 149, 385, 253, 541, 42,
     109, 190, 775, 906, 116, 761, 837, 452, 435, 410, 528, 172, 688, 8, 90, 394, 972, 134, 22, 883, 152, 386, 655, 820,
     366, 545, 103, 713, 170, 263, 819, 965, 322, 258, 647, 284, 274, 896, 901, 954, 305, 29, 216, 987, 615, 71, 607,
     83, 527, 56, 194, 412, 904, 510, 202, 978, 89, 213, 745, 842, 697, 40, 841, 104, 922, 827, 562, 138, 57, 709, 186,
     805, 184, 951, 780, 925, 429, 680, 848, 815, 160, 496, 39, 379, 6, 650, 245, 704, 244, 324, 755, 967, 276, 556,
     754, 716, 988, 217, 354, 87, 731, 751, 399, 117, 839, 369, 843, 871, 359, 727, 782, 64, 563, 532, 654, 240, 796,
     712, 569, 162, 331, 91, 559, 522, 648, 919, 383, 910, 60, 851, 914, 980, 131, 393, 762, 272, 13, 979, 437, 406,
     382, 999, 37, 728, 271, 898, 365, 924, 249, 223, 140, 401, 877, 808, 261, 857, 332, 88, 159, 642, 661, 759, 599,
     627, 788, 308, 730, 220, 858, 640, 540, 614, 127, 571, 329, 983, 188, 250, 26, 953, 909, 256, 789, 288, 446, 866,
     652, 918, 52, 241, 779, 593, 681, 863, 786, 504, 345, 872, 403, 351, 118, 835, 488, 892, 662, 595, 129, 33, 597,
     714, 441, 723, 503, 729, 873, 856, 692, 772, 426, 36, 77, 846, 416, 944, 62, 317, 594, 438, 800, 670, 431, 996,
     994, 285, 649, 119, 1, 405, 69, 47, 182, 123, 411, 707, 357, 377, 151, 286, 212, 575, 610, 853, 734, 497, 700, 834,
     899, 156, 428, 945, 894, 420, 178, 344, 870, 146, 302, 794, 70, 708, 51, 747, 622, 875, 998]) - 1


def ism(f, dim, ub, lb):
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
            p2 = copy.copy(p1)
            p2[i] = copy.copy(temp)
            fp2 = f(p2)
            fhat_archive[i] = fp2

        for j in range(i + 1, dim):
            counter += 1
            prev = prog
            prog = np.floor(counter / (dim * (dim - 1)) * 2 * 100)
            if prog % 5 == 0 and prog != prev:
                print("Progress: {}%".format(prog))
            if not np.isnan(fhat_archive[j]):
                fp3 = copy.copy(fhat_archive[j])
            else:
                p3 = copy.copy(p1)
                p3[j] = temp
                fp3 = f(p3)
                fhat_archive[j] = fp3
            p4 = copy.copy(p1)
            p4[i] = copy.copy(temp)
            p4[j] = copy.copy(temp)
            fp4 = f(p4)
            f_archive[i, j] = copy.copy(fp4)
            f_archive[j, i] = copy.copy(fp4)
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

    I1 = lambda_matrix <= errlb
    theta[I1] = 0
    I2 = lambda_matrix >= errub
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
    grayind = np.logical_and(lambda_matrix < errub, lambda_matrix > errlb)
    grayindsum = np.sum(grayind)
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


bench = Benchmark()
fun_fitness = bench.get_function(9)
info = bench.get_info(9)
dim = info['dimension']
sol = info['lower'] + np.random.rand(dim) * (info['upper'] - info['lower'])


class BayesEstimator():
    def __init__(self, dim):
        self.params = []
        self.dim = dim
        for i in range(self.dim):
            self.params.append([])
            for j in range(self.dim):
                if i == j:
                    self.params[i].append((1, 1))
                else:
                    self.params[i].append((1, 1))

    def __call__(self):
        y = np.zeros((self.dim, self.dim))
        for i in range(self.dim - 1):

            for j in range(i + 1, self.dim):
                a, b = self.params[i][j]
                y[i, j] = float(bernoulli.rvs(beta.rvs(a, b)))

        y = y + y.T
        return y

    def probs(self):
        y = np.zeros((self.dim, self.dim))
        for i in range(self.dim - 1):
            for j in range(i + 1, self.dim):
                a, b = self.params[i][j]
                y[i, j] = float(beta.mean(a, b))
        y = y + y.T
        return y

    def prob_random_given_X_geq_1(self, g1, g2):
        tuplas = []

        for i in range(len(g1)):
            for j in range(len(g2)):
                tuplas.append((g1[i], g2[j]))

        temp1 = 0
        for k in tuplas:
            temp1 *= (1 - beta.rvs(*self.params[k[0]][k[1]]))
            # for g in tuplas:
            #     if g[0] == k[0] and k[1] == g[1]:
            #         continue
            #     else:
            #         temp2 *= (1-beta.rvs(*self.params[g[0]][g[1]]))
            # temp1 += temp2
        return temp1

    def stochastic_update(self, g1, g2, bayes_estimator, real_matrix, ub, lb, dim):
        if isinstance(real_matrix, np.ndarray):

            experiment_result = []
            for elem1 in g1:
                for elem2 in g2:
                    experiment_result.append(real_matrix[elem1, elem2])
            if np.sum(experiment_result) > 0:
                P0 = self.prob_random_given_X_geq_1(g1, g2)
                for elem1 in g1:
                    for elem2 in g2:
                        a0, b0 = bayes_estimator.params[elem2][elem1]
                        p_of_1_given_X_grater = beta.rvs(a0, b0) / (1 - P0)
                        bayes_estimator.params[elem2][elem1] = (
                            a0 + p_of_1_given_X_grater, b0 + (1 - p_of_1_given_X_grater))
                        bayes_estimator.params[elem1][elem2] = (
                            a0 + p_of_1_given_X_grater, b0 + (1 - p_of_1_given_X_grater))

            else:

                for elem1 in g1:
                    for elem2 in g2:
                        if elem1 > elem2:
                            a0, b0 = bayes_estimator.params[elem2][elem1]
                            bayes_estimator.params[elem2][elem1] = (a0, b0 + 1)
                            bayes_estimator.params[elem1][elem2] = (a0, b0 + 1)
        else:

            temp = (ub + lb) / 2
            p1 = np.array([lb] * dim)
            p3 = copy.copy(p1)
            p2 = copy.copy(p1)
            p4 = copy.copy(p1)
            fp1 = real_matrix(p1)
            for i in g1:
                p2[i] = copy.copy(temp)
                p4[i] = copy.copy(temp)

            for j in g2:
                p3[j] = copy.copy(temp)
                p4[j] = copy.copy(temp)
            fp3 = real_matrix(p3)
            fp4 = real_matrix(p4)
            fp2 = real_matrix(p2)

            d1 = fp2 - fp1
            d2 = fp4 - fp3
            if abs(d1 - d2) > 1E-5:
                P0 = self.prob_random_given_X_geq_1(g1, g2)
                print("Calculated P0")
                for elem1 in g1:
                    for elem2 in g2:
                        a0, b0 = bayes_estimator.params[elem2][elem1]
                        p_of_1_given_X_grater = beta.rvs(a0, b0) / (1 - P0)
                        bayes_estimator.params[elem2][elem1] = (
                            a0 + p_of_1_given_X_grater, b0 + (1 - p_of_1_given_X_grater))
                        bayes_estimator.params[elem1][elem2] = (
                            a0 + p_of_1_given_X_grater, b0 + (1 - p_of_1_given_X_grater))


            else:
                for elem1 in g1:
                    for elem2 in g2:
                        if elem1 > elem2:
                            a0, b0 = bayes_estimator.params[elem2][elem1]
                            bayes_estimator.params[elem2][elem1] = (a0, b0 + 1)
                            bayes_estimator.params[elem1][elem2] = (a0, b0 + 1)


def random_grouping(set_of_eligible_indices, size1, size2):
    whole_group = np.random.choice(set_of_eligible_indices, size1 + size2, replace=False)
    g1 = whole_group[:size1]
    g2 = whole_group[size1:]
    return g1, g2


def update(g1, g2, bayes_estimator, real_matrix):
    experiment_result = []
    for elem1 in g1:
        for elem2 in g2:
            experiment_result.append(real_matrix[elem1, elem2])
    if np.sum(experiment_result) > 0:
        for elem1 in g1:
            for elem2 in g2:
                a0, b0 = bayes_estimator.params[elem2][elem1]
                bayes_estimator.params[elem2][elem1] = (a0 + 1, b0)
                bayes_estimator.params[elem1][elem2] = (a0 + 1, b0)

    else:

        for elem1 in g1:
            for elem2 in g2:
                if elem1 > elem2:
                    a0, b0 = bayes_estimator.params[elem2][elem1]
                    bayes_estimator.params[elem2][elem1] = (a0, b0 + 1)
                    bayes_estimator.params[elem1][elem2] = (a0, b0 + 1)


def recursive_adding(list_of_groups, list_of_ordered_indices):
    mitad = int(np.floor((len(list_of_groups) - 1) / 2))
    mitad_1 = list_of_ordered_indices[:mitad]
    mitad_2 = list_of_ordered_indices[:mitad]
    list_of_groups.append((copy.deepcopy(mitad_1), copy.deepcopy(mitad_2)))
    if len(mitad_1) > 1:
        recursive_adding(list_of_groups, mitad_1)
    elif len(mitad_2) > 1:
        recursive_adding(list_of_groups, mitad_2)


def iterative_splitting(list_of_groups, list_of_ordered_indices):
    max_exponent = int(np.log(len(list_of_ordered_indices)) // np.log(2))
    # max_number_of_groups = 2**max_exponent

    for exponent in range(1, max_exponent):
        group_number = 2 ** exponent
        elements_per_group = len(list_of_ordered_indices) // group_number
        j = 0
        temp = []
        while j < len(list_of_ordered_indices):
            try:
                temp.append(copy.deepcopy(list_of_ordered_indices[j:j + elements_per_group]))
                j += elements_per_group
            except:
                temp.append(copy.deepcopy(list_of_ordered_indices[j:]))
                j += elements_per_group

        list_of_groups.append(tuple(temp))


def secuential_grouping(list_of_groups, list_of_ordered_indices):
    i = 0
    while i < len(list_of_ordered_indices) - 1:
        list_of_groups.append(([list_of_ordered_indices[i]], [list_of_ordered_indices[i + 1]]))
        i += 2


def delta_grouping(dim, func, lb, ub, options, indices):
    population = options["population"]
    type_selection = options["selection"]
    # pop = np.random.uniform(lb, ub, (2 * population, dim))
    # values = []
    # for elem in pop:
    #     values.append(func(elem))
    # ordenada = np.argsort(values)
    #
    # mitad_alta = pop[ordenada[:population], :]
    # mitad_baja = pop[ordenada[population:], :]
    # diferencia = mitad_alta - mitad_baja
    #
    # delta_vector = np.mean(diferencia, axis=0)
    #
    # sort_indices = np.argsort(delta_vector)
    #
    list_of_groups = []
    temp = np.random.choice(range(len(indices)), 1)
    sort_indices = indices[int(temp)]
    if type_selection == "sequential":
        secuential_grouping(list_of_groups, sort_indices)
    elif type_selection == "merge":
        iterative_splitting(list_of_groups, sort_indices)
    elif type_selection == "further":
        further_splitting(list_of_groups, sort_indices)
    else:
        raise Exception("type selection \"{}\" is not suported.".format(type_selection))

    return list_of_groups


def get_partitions(array, k):
    jump = int(len(array) // k)

    j = 0
    temp = []
    while j < len(array):
        try:
            temp.append(copy.copy(array[j:j + jump]))
            j += jump
        except:
            original = temp[-1]
            new = np.stack((original, copy.copy(list_of_ordered_indices[j:])))
            temp[-1] = new
            j += jump
    return temp


def further_splitting(list_of_groups, sort_indices):
    sort_indices = np.squeeze(sort_indices)
    size_of_groups = (len(sort_indices) ** (1 / 2)) // 1

    groups = get_partitions(sort_indices, size_of_groups)
    if len(groups) == 1:
        pdb.set_trace()

    middle_of_groups = (len(groups) // 2) - 1
    if len(groups) % 2 == 0:
        i = 0
        while i <= middle_of_groups:
            try:
                obj1 = copy.deepcopy(groups[i])
                obj2 = copy.deepcopy(groups[i + middle_of_groups])
                list_of_groups.append((np.squeeze(obj1), np.squeeze(obj2)))
                i += 1
            except:
                print("Index {} out of bounds for array of length {}".format(i + middle_of_groups, len(groups)))

    else:
        i = 0
        while i <= middle_of_groups + 1:
            try:
                obj1 = copy.deepcopy(groups[i])
                obj2 = copy.deepcopy(groups[i + middle_of_groups])
                list_of_groups.append([np.squeeze(obj1), np.squeeze(obj2)])
                i += 1
            except:
                print("Index {} out of bounds for array of length {}".format(i + middle_of_groups, len(groups)))


def run_inferece(grouping="random", type_inference="naive", number_of_dimensions=15, iterations=100, function=None,
                 lb=0,
                 ub=0, ism=None, dsm=None, options_for_delta=None):
    if grouping == "random":
        if type_inference == "naive":

            N = number_of_dimensions

            actual_interaction = function
            # actual_interaction = actual_interaction + actual_interaction.T
            lambda_matrix, fhat_archive, f_archive, fp1 = ism(function, number_of_dimensions, ub, lb)
            non_seps, seps, theta, epsilon = dsm(fhat_archive, lambda_matrix, f_archive, fp1, number_of_dimensions)
            estimator = BayesEstimator(N)

            for iter in range((N ** 2 + N + 1) // 8):
                s1 = np.random.choice(range(1, int(N / 2)))
                s2 = np.random.choice(range(1, N - s1))
                g1, g2 = random_grouping(range(0, N), s1, s2)
                update(g1, g2, estimator, theta)
                if iter % 10 == 0:
                    print("Current function evaluations: {}/{}".format(iter, (N ** 2 + N + 1) // 8))
            trail = []
            probas = estimator.probs()
            for i in range(10):
                sample = estimator()
                r1, r2, r3 = rho_metrics(sample, theta)
                trail.append([r1, r2, r3])
            rho = np.mean(trail, axis=0)

            print("r1:{} r2:{}  r3:{}".format(rho[0], rho[1], rho[2]))

            plt.matshow(theta)
            plt.title(r"Real Matrix $\Theta$")
            plt.colorbar()
            plt.matshow(probas)
            plt.title("Mean of probability")
            plt.colorbar()
            plt.matshow(sample)
            plt.title("Bernoulli variable")
            plt.colorbar()
        else:

            N = number_of_dimensions
            actual_interaction = function
            # actual_interaction = actual_interaction + actual_interaction.T
            # actual_interaction[np.diag_indices_from(actual_interaction)]=1

            estimator = BayesEstimator(N)
            lambda_matrix, fhat_archive, f_archive, fp1 = ism(function, number_of_dimensions, ub, lb)
            non_seps, seps, theta, epsilon = dsm(fhat_archive, lambda_matrix, f_archive, fp1, number_of_dimensions)
            # This number is the number of evaluations done by DG2. each 2 groups comparison is 4 function calls
            for iter in range((N ** 2 + N + 1) // 8):
                s1 = np.random.choice(range(1, int(N / 2)))
                s2 = np.random.choice(range(1, N - s1))
                g1, g2 = random_grouping(range(0, N), s1, s2)
                if iter % 10 == 0:
                    print("Current function evaluations: {}/{}".format(iter * 4, (N ** 2 + N + 1) // 8))
                estimator.stochastic_update(g1, g2, estimator, actual_interaction, ub, lb, number_of_dimensions)
                # update(g1, g2, estimator, actual_interaction)
                if iter % 10 == 0:
                    print("Current function evaluations: {}/{}".format(iter * 4, (N ** 2 + N + 1) // 8))

            probas = estimator.probs()
            trail = []
            for i in range(10):
                sample = estimator()
                r1, r2, r3 = rho_metrics(sample, theta)
                trail.append([r1, r2, r3])
            rho = np.mean(trail, axis=0)

            print("r1:{} r2:{}  r3:{}".format(rho[0], rho[1], rho[2]))

            plt.matshow(theta)
            plt.title(r"Real Matrix $\Theta$")
            plt.colorbar()
            plt.matshow(probas)
            plt.title("Mean of probability")
            plt.colorbar()
            plt.matshow(sample)
            plt.title("Bernoulli variable")
            plt.colorbar()
            plt.show()
    else:
        if type_inference == "naive":
            assert function != None, "For grouping different from random the parameter `function` must not be None"
            assert options_for_delta != None, "You cant use delta grouping without specifying its options"
            assert lb != ub, "For grouping different from random the parameter `lb` must be different from " \
                             "`ub`"
            N = number_of_dimensions

            actual_interaction = function
            # actual_interaction = actual_interaction + actual_interaction.T

            estimator = BayesEstimator(N)
            # from DG2 import ism,dsm
            lambda_matrix, fhat_archive, f_archive, fp1 = ism(function, number_of_dimensions, ub, lb)
            non_seps, seps, theta, epsilon = dsm(fhat_archive, lambda_matrix, f_archive, fp1, number_of_dimensions)

            # This number is the number of evaluations done by DG2. each 2 groups comparison is 4 function calls
            max_function_calls = (N ** 2 + N + 1) // 8
            f = 0
            indices = run_indexes_from_delta_grouping(number_of_dimensions, function, lb, ub, options_for_delta[
                "population"])
            list_of_groups = delta_grouping(number_of_dimensions, function, lb, ub, options=options_for_delta,
                                            indices=indices)

            if options_for_delta["selection"] == "further":

                while f < max_function_calls:
                    for _ in indices:
                        i = 0
                        list_of_groups = delta_grouping(number_of_dimensions, function, lb, ub,
                                                        options=options_for_delta,
                                                        indices=indices)
                        while i < len(list_of_groups) and f < max_function_calls:
                            elem = list_of_groups[i]

                            # If this is squential or further version
                            if len(elem) == 2:

                                # pdb.set_trace()
                                update(elem[0].reshape(-1), elem[1].reshape(-1), estimator, theta)
                                f += 4
                            # If this is merge version (except the first group)
                            else:
                                n = 0

                                while n < len(elem) - 1 and f < max_function_calls:
                                    thing = (elem[n], elem[n + 1])
                                    update(thing[0], thing[1], estimator, theta)
                                    n += 2
                                    f += 4
                            print("Current function calls: {}/{}".format(f, max_function_calls))
                            i += 1
                        if i >= len(list_of_groups):
                            break

            else:
                while f < max_function_calls:
                    i = 0
                    while i < len(list_of_groups) and f < max_function_calls:
                        elem = list_of_groups[i]

                        # If this is squential version
                        if len(elem) == 2:
                            update(elem[0], elem[1], estimator, theta)
                            f += 4
                        # If this is merge version (except the first group)
                        else:
                            n = 0

                            while n < len(elem) - 1 and f < max_function_calls:
                                thing = (elem[n], elem[n + 1])
                                update(thing[0], thing[1], estimator, theta)
                                n += 2
                                f += 4
                        print("Current function calls: {}/{}".format(f, max_function_calls))
                        i += 1
                    if i >= len(list_of_groups):
                        break
            probas = estimator.probs()
            trail = []
            for i in range(10):
                sample = estimator()
                r1, r2, r3 = rho_metrics(sample, theta)
                trail.append([r1, r2, r3])
            rho = np.mean(trail, axis=0)

            print("r1:{} r2:{}  r3:{}".format(rho[0], rho[1], rho[2]))

            plt.matshow(theta)
            plt.title(r"Real Matrix $\Theta$")
            plt.colorbar()
            plt.matshow(probas)
            plt.title("Mean of probability")
            plt.colorbar()
            plt.matshow(sample)
            plt.title("Bernoulli variable")
            plt.colorbar()

            plt.show()
        else:
            assert function != None, "For grouping different from random the parameter `funtcion` must not be None"
            assert lb != ub, "For grouping different from random the parameter `lb` must be different from " \
                             "`ub`"

            N = number_of_dimensions

            actual_interaction = function

            estimator = BayesEstimator(N)
            # from DG2 import ism, dsm
            lambda_matrix, fhat_archive, f_archive, fp1 = ism(function, number_of_dimensions, ub, lb)
            non_seps, seps, theta, epsilon = dsm(fhat_archive, lambda_matrix, f_archive, fp1, number_of_dimensions)

            # This number is the number of evaluations done by DG2. each 2 groups comparison is 4 function calls
            max_function_calls = (N ** 2 + N + 1) // 8
            f = 0
            indices = run_indexes_from_delta_grouping(number_of_dimensions, function, lb, ub, options_for_delta[
                "population"])
            list_of_groups = delta_grouping(number_of_dimensions, function, lb, ub, options=options_for_delta,
                                            indices=indices)

            if options["selection"] == "further":

                while f < max_function_calls:
                    for _ in indices:
                        i = 0
                        list_of_groups = delta_grouping(number_of_dimensions, function, lb, ub,
                                                        options=options_for_delta,
                                                        indices=indices)
                        while i < len(list_of_groups) and f < max_function_calls:
                            elem = list_of_groups[i]

                            # If this is squential or further version
                            if len(elem) == 2:
                                update(elem[0], elem[1], estimator, theta)
                                f += 4
                            # If this is merge version (except the first group)
                            else:
                                n = 0

                                while n < len(elem) - 1 and f < max_function_calls:
                                    thing = (elem[n], elem[n + 1])
                                    update(thing[0], thing[1], estimator, theta)
                                    n += 2
                                    f += 4
                            print("Current function calls: {}/{}".format(f, max_function_calls))
                            i += 1
                        if i >= len(list_of_groups):
                            break

            else:
                while f < max_function_calls:
                    i = 0
                    while i < len(list_of_groups) and f < max_function_calls:
                        elem = list_of_groups[i]

                        # If this is squential version
                        if len(elem) == 2:
                            estimator.stochastic_update(elem[0], elem[1], estimator, actual_interaction, ub, lb,
                                                        number_of_dimensions)
                            f += 4
                        # If this is merge version (except the first group)
                        else:
                            n = 0

                            while n < len(elem) - 1 and f < max_function_calls:
                                thing = (elem[n], elem[n + 1])
                                estimator.stochastic_update(thing[0], thing[1], estimator, actual_interaction, ub, lb,
                                                            number_of_dimensions)
                                n += 2
                                f += 4
                        print("Current function calls: {}/{}".format(f, max_function_calls))
                        i += 1
                    if i >= len(list_of_groups):
                        break
                # update(g1, g2, estimator, theta)
            probas = estimator.probs()
            trail = []
            for i in range(10):
                sample = estimator()
                r1, r2, r3 = rho_metrics(sample, theta)
                trail.append([r1, r2, r3])
            rho = np.mean(trail, axis=0)

            print("r1:{} r2:{}  r3:{}".format(rho[0], rho[1], rho[2]))

            plt.matshow(theta)
            plt.title(r"Real Matrix $\Theta$")
            plt.colorbar()
            plt.matshow(probas)
            plt.title("Mean of probability")
            plt.colorbar()
            plt.matshow(sample)
            plt.title("Bernoulli variable")
            plt.colorbar()

            plt.show()


def mutation(x):
    indice = np.random.choice(range(len(x)))
    x[indice] += np.random.uniform(-1, 1)


def cross_over(x1, x2):
    indice = np.random.choice(range(len(x1)))
    hijo_1 = np.squeeze(np.hstack((x1[:indice].reshape(1, -1), x2[indice:].reshape(1, -1))))
    hijo_2 = np.squeeze(np.hstack((x2[:indice].reshape(1, -1), x1[indice:].reshape(1, -1))))
    return hijo_2, hijo_1


def optimization_step(pop, f):
    mitad = len(pop) // 2
    l = []

    for elem in pop:
        l.append(f(np.squeeze(elem)))
    fitness_for_pop = np.array(l)
    ordenados = np.argsort(fitness_for_pop)
    mejores = pop[ordenados[:mitad], :]
    nueva_pop = []
    while len(nueva_pop) < len(pop):
        p1, p2 = np.random.choice(range(len(mejores)), 2)
        padre1, madre1 = mejores[p1, :], mejores[p2, :]
        hijo1, hijo2 = cross_over(padre1, madre1)
        if np.random.rand() < 0.1:
            mutation(hijo1)
            mutation(hijo2)
        nueva_pop.append(hijo1)
        nueva_pop.append(hijo2)
    return np.array(nueva_pop)


def run_indexes_from_delta_grouping(dim, func, lb, ub, population, numer_of_iterations=100):
    pop = np.random.uniform(lb, ub, (population, dim))
    salida = []
    for i in range(numer_of_iterations):
        new_pop = optimization_step(pop, func)

        diferencia = new_pop - pop

        delta_vector = np.mean(diferencia, axis=0)

        sort_indices = np.argsort(delta_vector)
        salida.append(sort_indices)
        pop = copy.copy(new_pop)

    return salida


def rho_metrics(theta, real_theta):
    dim = theta.shape[0]
    theta = np.array(theta, dtype=np.int32)
    real_theta = np.array(real_theta, dtype=np.int32)
    resta = np.array(theta - real_theta)
    r1 = (np.sum(theta * real_theta - np.eye(dim)) / np.sum(real_theta - np.eye(dim))) * 100
    # pdb.set_trace()
    r2 = np.sum((np.ones((dim, dim)) - theta) * (np.ones((dim, dim)) - real_theta)) / np.sum(
        np.ones((dim, dim)) - real_theta) * 100
    r3 = np.sum(np.triu(np.ones((dim, dim)) - np.abs(resta)) - np.eye(dim)) * 2 / (dim * (dim - 1)) * 100

    return r1, r2, r3




