# -*- coding: utf-8 -*-
import numpy as np
import pickle
from nltk.corpus import twitter_samples
from utils import get_dict

with open("./data/test_cases.pkl", "rb") as test_file: test_cases_file = pickle.load(test_file)

def test_get_matrices(target):
    successful_cases = 0
    failed_cases = []

    en_fr_train = get_dict("./data/en-fr.train.txt")
    en_fr_test = get_dict("./data/en-fr.test.txt")
    en_embeddings_subset = pickle.load(open("./data/en_embeddings.pkl", "rb"))
    fr_embeddings_subset = pickle.load(open("./data/fr_embeddings.pkl", "rb"))

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "en_fr": en_fr_train,
                "french_vecs": fr_embeddings_subset,
                "english_vecs": en_embeddings_subset,
            },
            "expected": {
                "X_0": test_cases_file["test_get_matrices"]["default_check"]["X_0"],
                "Y_0": test_cases_file["test_get_matrices"]["default_check"]["Y_0"],
                "X_shape": (4932, 300),
                "Y_shape": (4932, 300),
                "X_last": test_cases_file["test_get_matrices"]["default_check"]["X_last"],
                "Y_last": test_cases_file["test_get_matrices"]["default_check"]["Y_last"]
            },
        },
        {
            "name": "test_check",
            "input": {
                "en_fr": en_fr_test,
                "french_vecs": fr_embeddings_subset,
                "english_vecs": en_embeddings_subset,
            },
            "expected": {
                "X_0": test_cases_file["test_get_matrices"]["test_check"]["X_0"],
                "Y_0": test_cases_file["test_get_matrices"]["test_check"]["Y_0"],
                "X_shape": (1438, 300),
                "Y_shape": (1438, 300),
                "X_last": test_cases_file["test_get_matrices"]["test_check"]["X_last"],
                "Y_last": test_cases_file["test_get_matrices"]["test_check"]["Y_last"],
            },
        },
    ]

    for test_case in test_cases:
        result_x, result_y = target(**test_case["input"])

        try:
            assert isinstance(result_x, np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": type(test_case["expected"]["X_0"]),
                    "got": type(result_x),
                }
            )
            print(
                f"Wrong output type for X matrix. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert isinstance(result_y, np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": type(test_case["expected"]["Y_0"]),
                    "got": type(result_y),
                }
            )
            print(
                f"Wrong output type for Y matrix. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result_x[0], test_case["expected"]["X_0"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["X_0"][:10],
                    "got": result_x[0][:10],
                }
            )
            print(
                f"Wrong output values for X matrix. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result_y[0], test_case["expected"]["Y_0"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["Y_0"][:10],
                    "got": result_y[0][:10],
                }
            )
            print(
                f"Wrong output values for Y matrix. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result_x.shape == test_case["expected"]["X_shape"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["X_shape"],
                    "got": result_x.shape,
                }
            )
            print(
                f"Wrong shape for X matrix. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result_y.shape == test_case["expected"]["Y_shape"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["Y_shape"],
                    "got": result_y.shape,
                }
            )
            print(
                f"Wrong shape for Y matrix. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result_x[-1], test_case["expected"]["X_last"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["X_last"][:10],
                    "got": result_x[-1][:10],
                }
            )
            print(
                f"Wrong output values for X matrix. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result_y[-1], test_case["expected"]["Y_last"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["Y_last"][:10],
                    "got": result_y[-1][:10],
                }
            )
            print(
                f"Wrong output values for Y matrix. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_compute_loss(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "X": np.array(
                    [
                        [0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897],
                        [0.42310646, 0.9807642, 0.68482974, 0.4809319, 0.39211752],
                        [0.34317802, 0.72904971, 0.43857224, 0.0596779, 0.39804426],
                        [0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759],
                        [0.63440096, 0.84943179, 0.72445532, 0.61102351, 0.72244338],
                        [0.32295891, 0.36178866, 0.22826323, 0.29371405, 0.63097612],
                        [0.09210494, 0.43370117, 0.43086276, 0.4936851, 0.42583029],
                        [0.31226122, 0.42635131, 0.89338916, 0.94416002, 0.50183668],
                        [0.62395295, 0.1156184, 0.31728548, 0.41482621, 0.86630916],
                        [0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453],
                    ]
                ),
                "Y": np.array(
                    [
                        [0.01206287, 0.08263408, 0.06030601, 0.0545068, 0.03427638],
                        [0.03041208, 0.04170222, 0.06813008, 0.08754568, 0.05104223],
                        [0.06693138, 0.05859366, 0.06249035, 0.06746891, 0.08423424],
                        [0.0083195, 0.07636828, 0.02436664, 0.0194223, 0.0572457],
                        [0.00957125, 0.08853268, 0.0627249, 0.07234164, 0.00161292],
                        [0.05944319, 0.05567852, 0.01589596, 0.01530705, 0.06955295],
                        [0.03187664, 0.06919703, 0.05543832, 0.03889506, 0.09251325],
                        [0.084167, 0.03573976, 0.00435915, 0.03047681, 0.03981857],
                        [0.07049588, 0.09953585, 0.03559149, 0.07625478, 0.05931769],
                        [0.06917018, 0.01511275, 0.03988763, 0.02408559, 0.0343456],
                    ]
                ),
                "R": np.array(
                    [
                        [0.51312815, 0.66662455, 0.10590849, 0.13089495, 0.32198061],
                        [0.66156434, 0.84650623, 0.55325734, 0.85445249, 0.38483781],
                        [0.3167879, 0.35426468, 0.17108183, 0.82911263, 0.33867085],
                        [0.55237008, 0.57855147, 0.52153306, 0.00268806, 0.98834542],
                        [0.90534158, 0.20763586, 0.29248941, 0.52001015, 0.90191137],
                    ]
                ),
            },
            "expected": 8.186626624823763,
        },
        {
            "name": "small_check",
            "input": {
                "X": np.array([[0.10]]),
                "Y": np.array([[0.20]]),
                "R": np.array([[0.30]]),
            },
            "expected": 0.028900000000000006,
        },
        {
            "name": "large_check",
            "input": {
                "X": np.array([[0.10, 0.1], [0.2, 0.4], [0.6, -0.2]]),
                "Y": np.array([[0.20, 0.4], [0.3, 0.5], [0.1, 0.0]]),
                "R": np.array([[0.30, -0.1], [-0.4, 0.2]]),
            },
            "expected": 0.19513333333333335,
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert np.isclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Wrong output loss. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_compute_gradient(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "X": np.array(
                    [
                        [0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897],
                        [0.42310646, 0.9807642, 0.68482974, 0.4809319, 0.39211752],
                        [0.34317802, 0.72904971, 0.43857224, 0.0596779, 0.39804426],
                        [0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759],
                        [0.63440096, 0.84943179, 0.72445532, 0.61102351, 0.72244338],
                        [0.32295891, 0.36178866, 0.22826323, 0.29371405, 0.63097612],
                        [0.09210494, 0.43370117, 0.43086276, 0.4936851, 0.42583029],
                        [0.31226122, 0.42635131, 0.89338916, 0.94416002, 0.50183668],
                        [0.62395295, 0.1156184, 0.31728548, 0.41482621, 0.86630916],
                        [0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453],
                    ]
                ),
                "Y": np.array(
                    [
                        [0.01206287, 0.08263408, 0.06030601, 0.0545068, 0.03427638],
                        [0.03041208, 0.04170222, 0.06813008, 0.08754568, 0.05104223],
                        [0.06693138, 0.05859366, 0.06249035, 0.06746891, 0.08423424],
                        [0.0083195, 0.07636828, 0.02436664, 0.0194223, 0.0572457],
                        [0.00957125, 0.08853268, 0.0627249, 0.07234164, 0.00161292],
                        [0.05944319, 0.05567852, 0.01589596, 0.01530705, 0.06955295],
                        [0.03187664, 0.06919703, 0.05543832, 0.03889506, 0.09251325],
                        [0.084167, 0.03573976, 0.00435915, 0.03047681, 0.03981857],
                        [0.07049588, 0.09953585, 0.03559149, 0.07625478, 0.05931769],
                        [0.06917018, 0.01511275, 0.03988763, 0.02408559, 0.0343456],
                    ]
                ),
                "R": np.array(
                    [
                        [0.51312815, 0.66662455, 0.10590849, 0.13089495, 0.32198061],
                        [0.66156434, 0.84650623, 0.55325734, 0.85445249, 0.38483781],
                        [0.3167879, 0.35426468, 0.17108183, 0.82911263, 0.33867085],
                        [0.55237008, 0.57855147, 0.52153306, 0.00268806, 0.98834542],
                        [0.90534158, 0.20763586, 0.29248941, 0.52001015, 0.90191137],
                    ]
                ),
            },
            "expected": np.array(
                [
                    [1.3498175, 1.11264981, 0.69626762, 0.98468499, 1.33828969],
                    [1.48402939, 1.3134471, 0.8269311, 1.27307285, 1.44181639],
                    [1.57868759, 1.3817686, 0.89039471, 1.35293657, 1.60202282],
                    [1.50368303, 1.27421294, 0.8258529, 1.16996514, 1.54811674],
                    [1.72780859, 1.41902443, 0.90765656, 1.31399276, 1.73329241],
                ]
            ),
        },
        {
            "name": "small_check",
            "input": {
                "X": np.array([[0.10]]),
                "Y": np.array([[0.20]]),
                "R": np.array([[0.30]]),
            },
            "expected": np.array([[-0.034]]),
        },
        {
            "name": "large_check",
            "input": {
                "X": np.array([[0.10, 0.1], [0.2, 0.4], [0.6, -0.2]]),
                "Y": np.array([[0.20, 0.4], [0.3, 0.5], [0.1, 0.0]]),
                "R": np.array([[0.30, -0.1], [-0.4, 0.2]]),
            },
            "expected": np.array([[-0.00333333, -0.12466667], [-0.142, -0.13]]),
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert result.shape == test_case["expected"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"].shape,
                    "got": result.shape,
                }
            )
            print(
                f"Wrong shape for gradient matrix. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Wrong output values for gradient matrix. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_align_embeddings(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "X": np.array(
                    [
                        [0.57053163, 0.26825978, 0.60529524, 0.08485638, 0.97365435],
                        [0.51633925, 0.88898933, 0.0973924, 0.08909037, 0.01842588],
                        [0.78425402, 0.02219408, 0.16195252, 0.42449592, 0.22851964],
                        [0.17774345, 0.86861078, 0.33408275, 0.80437326, 0.46245774],
                        [0.3755519, 0.64955325, 0.44143494, 0.16636594, 0.49207735],
                        [0.79337432, 0.09869404, 0.2810426, 0.39652654, 0.81000488],
                        [0.1523301, 0.37393728, 0.9452125, 0.59574785, 0.72800603],
                        [0.21238829, 0.32974222, 0.11960557, 0.18153431, 0.2164236],
                        [0.12539322, 0.96034526, 0.71677102, 0.19275559, 0.09343238],
                        [0.05759864, 0.07582363, 0.67205969, 0.30242147, 0.46522306],
                    ]
                ),
                "Y": np.array(
                    [
                        [0.0233235, 0.04843668, 0.08284301, 0.01447825, 0.04674289],
                        [0.00164, 0.03476209, 0.04132209, 0.02619716, 0.03767882],
                        [0.02736822, 0.0905283, 0.00564728, 0.06326545, 0.08689923],
                        [0.03768616, 0.07806676, 0.04601056, 0.04168253, 0.08640853],
                        [0.03573465, 0.06263811, 0.02605627, 0.09576452, 0.02241846],
                        [0.0690478, 0.05178494, 0.02649317, 0.01818275, 0.04840006],
                        [0.03277532, 0.02276209, 0.09912554, 0.00064027, 0.05491281],
                        [0.00834599, 0.05053472, 0.04568681, 0.02292083, 0.06566439],
                        [0.08895743, 0.09154556, 0.00046113, 0.08459362, 0.0608969],
                        [0.00460105, 0.0183352, 0.09005481, 0.01632413, 0.01361763],
                    ]
                ),
                "train_steps": 100,
                "learning_rate": 0.0003,
                "verbose": False,
            },
            "expected": np.array(
                [
                    [
                        5.49166091e-01,
                        2.46373373e-01,
                        5.89727712e-01,
                        7.37927874e-02,
                        9.52671617e-01,
                    ],
                    [
                        4.89061423e-01,
                        8.58166034e-01,
                        8.31315085e-02,
                        7.48037161e-02,
                        8.36581242e-04,
                    ],
                    [
                        7.56135061e-01,
                        -5.60399011e-03,
                        1.46376115e-01,
                        4.07797332e-01,
                        2.08062240e-01,
                    ],
                    [
                        1.58350913e-01,
                        8.46338260e-01,
                        3.21697639e-01,
                        7.91145852e-01,
                        4.46340717e-01,
                    ],
                    [
                        3.46947901e-01,
                        6.20500819e-01,
                        4.22710262e-01,
                        1.49351183e-01,
                        4.66880248e-01,
                    ],
                ]
            ),
        },
        {
            "name": "random_check_large_lr",
            "input": {
                "X": np.array(
                    [
                        [0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897],
                        [0.42310646, 0.9807642, 0.68482974, 0.4809319, 0.39211752],
                        [0.34317802, 0.72904971, 0.43857224, 0.0596779, 0.39804426],
                        [0.73799541, 0.18249173, 0.17545176, 0.53155137, 0.53182759],
                        [0.63440096, 0.84943179, 0.72445532, 0.61102351, 0.72244338],
                        [0.32295891, 0.36178866, 0.22826323, 0.29371405, 0.63097612],
                        [0.09210494, 0.43370117, 0.43086276, 0.4936851, 0.42583029],
                        [0.31226122, 0.42635131, 0.89338916, 0.94416002, 0.50183668],
                        [0.62395295, 0.1156184, 0.31728548, 0.41482621, 0.86630916],
                        [0.25045537, 0.48303426, 0.98555979, 0.51948512, 0.61289453],
                    ]
                ),
                "Y": np.array(
                    [
                        [0.01206287, 0.08263408, 0.06030601, 0.0545068, 0.03427638],
                        [0.03041208, 0.04170222, 0.06813008, 0.08754568, 0.05104223],
                        [0.06693138, 0.05859366, 0.06249035, 0.06746891, 0.08423424],
                        [0.0083195, 0.07636828, 0.02436664, 0.0194223, 0.0572457],
                        [0.00957125, 0.08853268, 0.0627249, 0.07234164, 0.00161292],
                        [0.05944319, 0.05567852, 0.01589596, 0.01530705, 0.06955295],
                        [0.03187664, 0.06919703, 0.05543832, 0.03889506, 0.09251325],
                        [0.084167, 0.03573976, 0.00435915, 0.03047681, 0.03981857],
                        [0.07049588, 0.09953585, 0.03559149, 0.07625478, 0.05931769],
                        [0.06917018, 0.01511275, 0.03988763, 0.02408559, 0.0343456],
                    ]
                ),
                "train_steps": 10,
                "learning_rate": 0.8,
                "verbose": False,
            },
            "expected": np.array(
                [
                    [1.49222726, 1.27201502, 0.99502832, 0.80565394, 1.40986691],
                    [1.42160903, 1.93442368, 0.97946232, 0.94203655, 1.21209401],
                    [1.8562409, 1.35159242, 1.07078329, 1.00669748, 1.46269503],
                    [1.31752657, 2.00481943, 0.96706243, 1.22178323, 1.28864589],
                    [1.64214155, 2.00758298, 1.07984998, 1.01051933, 1.33904183],
                ]
            ),
        },
        {
            "name": "smallest_check",
            "input": {
                "X": np.array([[0.10]]),
                "Y": np.array([[0.20]]),
                "train_steps": 10,
                "learning_rate": 0.01,
                "verbose": False,
            },
            "expected": np.array([[0.57338799]]),
        },
        {
            "name": "small_check",
            "input": {
                "X": np.array([[0.10, 0.1], [0.2, 0.4], [0.6, -0.2]]),
                "Y": np.array([[0.20, 0.4], [0.3, 0.5], [0.1, 0.0]]),
                "train_steps": 50,
                "learning_rate": 0.005,
                "verbose": False,
            },
            "expected": np.array([[0.55832269, 0.2735995], [0.60690436, 0.12258995]]),
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert result.shape == test_case["expected"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"].shape,
                    "got": result.shape,
                }
            )
            print(
                f"Wrong shape for rotation matrix after applying gradient descent. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Wrong output values for rotation matrix after gradient descent. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_nearest_neighbor(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "v": np.array([1, 0, 1]),
                "candidates": np.array(
                    [[1, 0, 5], [-2, 5, 3], [2, 0, 1], [6, -9, 5], [9, 9, 9]]
                ),
                "k": 3,
            },
            "expected": np.array([2, 0, 4]),
        },
        {
            "name": "larger_check",
            "input": {
                "v": np.array([1, 0, -1, 1, 0]),
                "candidates": np.array(
                    [
                        [2, 1, 0, 5, 0],
                        [-2, 5, 3, 1, -1],
                        [2, 0, 1, 0, 0],
                        [6, -9, 5, -3, 8],
                        [9, 9, 9, 9, 9],
                        [-8, -8, -8, -8, -8],
                        [2, 0, -5, 3, 0],
                    ]
                ),
                "k": 4,
            },
            "expected": np.array([6, 0, 4, 2]),
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert result.shape == (test_case["input"]["k"],)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": (test_case["input"]["k"],),
                    "got": result.shape,
                }
            )
            print(
                f"Wrong output shape. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Wrong ids were returned from nearest neighbor function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def unittest_test_vocabulary(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "small_check",
            "input": {
                "X": np.array([[0.10], [0.10]]),
                "Y": np.array([[0.20], [0.20]]),
                "R": np.array([[0.30]]),
            },
            "expected": 0.5,
        },
        {
            "name": "another_check",
            "input": {
                "X": np.array([[0.10, 0.1], [0.2, 0.4], [0.6, -0.2], [0.6, -0.5]]),
                "Y": np.array([[0.20, 0.4], [0.3, 0.5], [0.1, 0.0], [-0.2, 0.8]]),
                "R": np.array([[0.30, -0.1], [-0.4, 0.2]]),
            },
            "expected": 0.25,
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])
        try:
            assert np.isclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Wrong accuracy. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_get_document_embedding(target):
    successful_cases = 0
    failed_cases = []

    en_embeddings_subset = pickle.load(open("./data/en_embeddings.pkl", "rb"))

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "tweet": "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np",
                "en_embeddings": en_embeddings_subset,
            },
            "expected": test_cases_file["test_get_document_embedding"]["default_check"]["expected"],
        },
        {
            "name": "positive_check",
            "input": {
                "tweet": "Thank you :-))) OK \nHave fun!!\n@anvy2446 @4HUMANITEEs @SexyAF12 @kikbella @adasamper @RachelLFilsoof\nEnjoy day u all\nhttp://t.co/5Y5OAESAzv",
                "en_embeddings": en_embeddings_subset,
            },
            "expected": test_cases_file["test_get_document_embedding"]["positive_check"]["expected"],
        },
        {
            "name": "negative_check",
            "input": {
                "tweet": "@f0ggstar @stuartthull work neighbour on motors. Asked why and he said hates the updates on search :( http://t.co/XvmTUikWln",
                "en_embeddings": en_embeddings_subset,
            },
            "expected": test_cases_file["test_get_document_embedding"]["negative_check"]["expected"],
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert result.shape == test_case["expected"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"].shape,
                    "got": result.shape,
                }
            )
            print(
                f"Wrong output shape. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result, test_case["expected"], atol=1e-05
            )  # np.allclose(result[:10], test_case["expected"][:10])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"][:10],
                    "got": result[:10],
                }
            )
            print(
                f"Wrong document embedding. Remember to use 0 if the word is not in the embeddings. \n\tExpected first 10 elements: {failed_cases[-1].get('expected')}.\n\tGot those first 10 elements: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result, test_case["expected"], atol=1e-05
            )  # np.allclose(result[-10:], test_case["expected"][-10:])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"][-10:],
                    "got": result[-10:],
                }
            )
            print(
                f"Wrong document embedding. Remember to use 0 if the word is not in the embeddings. \n\tExpected first 10 elements: {failed_cases[-1].get('expected')}.\n\tGot those first 10 elements: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_get_document_vecs(target):
    successful_cases = 0
    failed_cases = []

    en_embeddings_subset = pickle.load(open("./data/en_embeddings.pkl", "rb"))
    all_positive_tweets = twitter_samples.strings("positive_tweets.json")
    all_negative_tweets = twitter_samples.strings("negative_tweets.json")
    all_tweets = all_positive_tweets + all_negative_tweets

    del all_positive_tweets, all_negative_tweets

    test_cases = [
        {
            "name": "default_check",
            "input": {"all_docs": all_tweets, "en_embeddings": en_embeddings_subset},
            "expected": {
                "len_dict": 10000,
                "docs_shape": (10000, 300),
                "doc_vec_0_0until10": np.array(
                    [
                        0.04821777,
                        -0.01904297,
                        0.15820312,
                        -0.18847656,
                        -0.11352539,
                        -0.1640625,
                        -0.1550293,
                        -0.63964844,
                        0.52001953,
                        0.04882812,
                    ]
                ),
                "doc_vec_last_0until10": np.array(
                    [
                        -0.30578613,
                        0.3984375,
                        -0.17480469,
                        -0.33544922,
                        -0.19067383,
                        -0.02441406,
                        0.00244141,
                        -0.44140625,
                        0.88818359,
                        0.37170401,
                    ]
                ),
                "ind2tweet_0_0until10": np.array(
                    [
                        0.04821777,
                        -0.01904297,
                        0.15820312,
                        -0.18847656,
                        -0.11352539,
                        -0.1640625,
                        -0.1550293,
                        -0.63964844,
                        0.52001953,
                        0.04882812,
                    ]
                ),
            },
        },
        {
            "name": "small_check",
            "input": {
                "all_docs": [
                    "@f0ggstar @stuartthull work neighbour on motors. Asked why and he said hates the updates on search :( http://t.co/XvmTUikWln",
                    "Thank you :-))) OK \nHave fun!!\n@anvy2446 @4HUMANITEEs @SexyAF12 @kikbella @adasamper @RachelLFilsoof\nEnjoy day u all\nhttp://t.co/5Y5OAESAzv",
                    "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np",
                ],
                "en_embeddings": en_embeddings_subset,
            },
            "expected": {
                "len_dict": 3,
                "docs_shape": (3, 300),
                "doc_vec_0_0until10": np.array(
                    [
                        0.22283936,
                        0.28491211,
                        0.25024414,
                        0.04541016,
                        -0.68408203,
                        0.09716797,
                        0.62237549,
                        -1.06835938,
                        0.04943848,
                        0.42053223,
                    ]
                ),
                "doc_vec_last_0until10": np.array(
                    [
                        -0.09228516,
                        0.35986328,
                        -0.0206604,
                        0.63085938,
                        -0.06640625,
                        -0.16308594,
                        0.12438965,
                        -0.38378906,
                        0.2052002,
                        0.72949219,
                    ]
                ),
                "ind2tweet_0_0until10": np.array(
                    [
                        0.22283936,
                        0.28491211,
                        0.25024414,
                        0.04541016,
                        -0.68408203,
                        0.09716797,
                        0.62237549,
                        -1.06835938,
                        0.04943848,
                        0.42053223,
                    ]
                ),
            },
        },
    ]

    for test_case in test_cases:
        result_doc, result_dict = target(**test_case["input"])

        try:
            assert result_doc.shape == test_case["expected"]["docs_shape"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["docs_shape"],
                    "got": result_doc.shape,
                }
            )
            print(
                f"Wrong output shape for document_vec_matrix. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert len(result_dict) == test_case["expected"]["len_dict"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["len_dict"],
                    "got": len(result_dict),
                }
            )
            print(
                f"Wrong number of elements in the ind2Doc_dict dictionary. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result_doc[0][:10],
                test_case["expected"]["doc_vec_0_0until10"],
                atol=1e-05,
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["doc_vec_0_0until10"],
                    "got": result_doc[0][:10],
                }
            )
            print(
                f"Wrong document embedding. \n\tExpected first 10 elements: {failed_cases[-1].get('expected')}.\n\tGot those first 10 elements: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result_doc[-1][:10],
                test_case["expected"]["doc_vec_last_0until10"],
                atol=1e-05,
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["doc_vec_last_0until10"],
                    "got": result_doc[-1][:10],
                }
            )
            print(
                f"Wrong document embedding. \n\tExpected first 10 elements: {failed_cases[-1].get('expected')}.\n\tGot those first 10 elements: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result_doc[-1][:10],
                test_case["expected"]["doc_vec_last_0until10"],
                atol=1e-05,
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["doc_vec_last_0until10"],
                    "got": result_doc[-1][:10],
                }
            )
            print(
                f"Wrong document embedding. \n\tExpected first 10 elements: {failed_cases[-1].get('expected')}.\n\tGot those first 10 elements: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result_dict[0][:10],
                test_case["expected"]["ind2tweet_0_0until10"],
                atol=1e-05,
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["ind2tweet_0_0until10"],
                    "got": result_dict[0][:10],
                }
            )
            print(
                f"Wrong embedding in dictionary at index 0. \n\tExpected first 10 elements: {failed_cases[-1].get('expected')}.\n\tGot those first 10 elements: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_hash_value_of_vector(target):
    successful_cases = 0
    failed_cases = []

    #  default_check
    N_DIMS = 300
    N_PLANES = 10
    N_UNIVERSES = 25

    np.random.seed(0)
    planes_l_default = [
        np.random.normal(size=(N_DIMS, N_PLANES)) for _ in range(N_UNIVERSES)
    ]
    idx = 0
    planes_default = planes_l_default[idx]

    np.random.seed(0)
    vec = np.random.rand(1, 300)

    # small_check
    N_DIMS = 20
    N_PLANES = 8
    N_UNIVERSES = 5

    np.random.seed(0)
    planes_l_small = [
        np.random.normal(size=(N_DIMS, N_PLANES)) for _ in range(N_UNIVERSES)
    ]
    idx = 4
    planes_small = planes_l_small[idx]

    np.random.seed(0)
    vec_small = np.random.rand(1, 20)

    # small_check2
    N_DIMS = 20
    N_PLANES = 8
    N_UNIVERSES = 5

    np.random.seed(0)
    planes_l_small = [
        np.random.normal(size=(N_DIMS, N_PLANES)) for _ in range(N_UNIVERSES)
    ]
    idx = 0
    planes_small2 = planes_l_small[idx]

    np.random.seed(0)
    vec_small2 = np.random.rand(1, 20)

    test_cases = [
        {
            "name": "default_check",
            "input": {"v": vec, "planes": planes_default},
            "expected": 768,
        },
        {
            "name": "small_check",
            "input": {"v": vec_small, "planes": planes_small},
            "expected": 70,
        },
        {
            "name": "small_check2",
            "input": {"v": vec_small2, "planes": planes_small2},
            "expected": 251,
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert result == test_case["expected"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Wrong hash value. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_make_hash_table(target):
    successful_cases = 0
    failed_cases = []

    np.random.seed(0)
    planes_l = [np.random.normal(size=(20, 8)) for _ in range(5)]

    test_cases = [
        {
            "name": "small_check1",
            "input": {"vecs": test_cases_file["test_make_hash_table"]["doc_vecs_test"], "planes": planes_l[0]},
            "expected": {
                "len_tables": 256,
                "id_to_eval": 225,
                "id_table": [3, 5],
                "hash_table": [
                    np.array(
                        [
                            -0.10229492,
                            0.0703125,
                            0.03009033,
                            0.24853516,
                            -0.21325684,
                            -0.05712891,
                            0.29541016,
                            -0.07727051,
                            0.32373047,
                            -0.07897949,
                            -0.00805664,
                            -0.25390625,
                            -0.07424927,
                            -0.03613281,
                            -0.36523438,
                            0.26170349,
                            0.04815674,
                            0.33496094,
                            0.06591797,
                            -0.29589844,
                        ]
                    ),
                    np.array(
                        [
                            -0.19726562,
                            0.21289062,
                            -0.02746582,
                            0.01013184,
                            -0.08007812,
                            -0.03320312,
                            -0.0291748,
                            -0.08300781,
                            0.08837891,
                            -0.12695312,
                            -0.13085938,
                            -0.11865234,
                            -0.53515625,
                            0.24902344,
                            -0.25585938,
                            0.28320312,
                            -0.18554688,
                            0.05419922,
                            0.0480957,
                            -0.20019531,
                        ]
                    ),
                ],
            },
        },
        {
            "name": "small_check2",
            "input": {"vecs": test_cases_file["test_make_hash_table"]["doc_vecs_test"], "planes": planes_l[4]},
            "expected": {
                "len_tables": 256,
                "id_to_eval": 182,
                "id_table": [6, 9],
                "hash_table": [
                    np.array(
                        [
                            0.1262207,
                            0.38085938,
                            0.00390625,
                            -0.33935547,
                            0.20483398,
                            -0.28515625,
                            0.25219727,
                            -0.01171875,
                            0.25622559,
                            0.26416016,
                            0.15136719,
                            -0.11096191,
                            0.13256836,
                            -0.0078125,
                            0.07617188,
                            -0.11405087,
                            -0.05322266,
                            0.16259766,
                            -0.05410767,
                            0.16650391,
                        ]
                    ),
                    np.array(
                        [
                            -0.30578613,
                            0.3984375,
                            -0.17480469,
                            -0.33544922,
                            -0.19067383,
                            -0.02441406,
                            0.00244141,
                            -0.44140625,
                            0.88818359,
                            0.3717041,
                            0.01751709,
                            -0.02783203,
                            -0.08496094,
                            -0.23486328,
                            0.17553711,
                            0.35400391,
                            -0.28955078,
                            0.09989929,
                            -0.28662109,
                            -0.18212891,
                        ]
                    ),
                ],
            },
        },
    ]

    for test_case in test_cases:
        result_hash, result_id = target(**test_case["input"])

        try:
            assert len(result_hash) == test_case["expected"]["len_tables"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["len_tables"],
                    "got": len(result_hash),
                }
            )
            print(
                f"Wrong number of elements in hash table. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert len(result_id) == test_case["expected"]["len_tables"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["len_tables"],
                    "got": len(result_hash),
                }
            )
            print(
                f"Wrong number of elements in id table. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert (
                result_id[test_case["expected"]["id_to_eval"]]
                == test_case["expected"]["id_table"]
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["id_table"],
                    "got": result_id[test_case["expected"]["id_to_eval"]],
                }
            )
            print(
                f"Wrong ids value at index {test_case['expected']['id_to_eval']} in id table. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result_hash[test_case["expected"]["id_to_eval"]],
                test_case["expected"]["hash_table"],
                atol=1e-05,
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["hash_table"],
                    "got": result_hash[test_case["expected"]["id_to_eval"]],
                }
            )
            print(
                f"Wrong ids value at index {test_case['expected']['id_to_eval']} in id table. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_approximate_knn(target, hash_tables, id_tables):
    successful_cases = 0
    failed_cases = []

    N_DIMS = 300
    N_PLANES = 10
    N_UNIVERSES = 25

    np.random.seed(0)
    planes_l = [np.random.normal(size=(N_DIMS, N_PLANES)) for _ in range(N_UNIVERSES)]

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "doc_id": 0,
                "v": test_cases_file["test_approximate_knn"]["default_check"]["v"],
                "planes_l": planes_l,
                "k": 3,
                "num_universes_to_use": 5,
                "hash_tables": hash_tables,
                "id_tables": id_tables,
            },
            "expected": [51, 2478, 105],
        },
        {
            "name": "default_check2",
            "input": {
                "doc_id": 5,
                "v": test_cases_file["test_approximate_knn"]["default_check2"]["v"],
                "planes_l": planes_l,
                "k": 4,
                "num_universes_to_use": 6,
                "hash_tables": hash_tables,
                "id_tables": id_tables,
            },
            "expected": [992, 4750, 9486, 9101],
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert result == test_case["expected"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Wrong chosen neighbor ids. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases
