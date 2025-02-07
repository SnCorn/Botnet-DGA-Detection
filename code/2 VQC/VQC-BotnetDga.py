
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import SPSA, ADAM, AQGD, CG, COBYLA, L_BFGS_B, GSLS, NELDER_MEAD, NFT, P_BFGS, POWELL, SLSQP, TNC
from qiskit.aqua.components.feature_maps import RawFeatureVector
from qiskit.circuit.library import TwoLocal, PauliFeatureMap, ZFeatureMap, ZZFeatureMap, NLocal, TwoLocal, RealAmplitudes, EfficientSU2, ExcitationPreserving
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name, get_feature_dimension
import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import datetime
import concurrent.futures
import time

def load_data(filepath):

    with open(filepath) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=np.int)

    return data, target, target_names


def _convert_data_dataframe(data, target,
                            feature_names, target_names):
    data_df = pd.DataFrame(data, columns=feature_names)
    target_df = pd.DataFrame(target, columns=target_names)
    combined_df = pd.concat([data_df, target_df], axis=1)
    X = combined_df[feature_names]
    y = combined_df[target_names]
    if y.shape[1] == 1:
        y = y.iloc[:, 0]
    return combined_df, X, y


def load_botnetdga(*, as_frame=False):

    data, target, target_names = load_data('dataset/BotnetDgaDataset.csv')

    with open('dataset/BotnetDgaDataset.rst') as rst_file:
        fdescr = rst_file.read()

    feature_names = ['MinREBotnets',
                     'CharLength',
                     'TreeNewFeature',
                     'nGramReputation_Alexa']

    frame = None
    target_columns = ['target', ]
    if as_frame:
        frame, data, target = _convert_data_dataframe(data,
                                                      target,
                                                      feature_names,
                                                      target_columns)

    return data, target


def botnetdga(training_size, test_size, n,
              standardize=False, pca=False, scale=False, plot_data=False,
              binarize=False):

    class_labels = [r'benign', r'dga']

    data, target = load_botnetdga()
    sample_train, sample_test, label_train, label_test = \
        train_test_split(data, target, train_size=training_size, test_size=test_size, random_state=7)

    # print("\n load_botnetdga sample_train = \n")
    # print(sample_train)
    # print("\n load_botnetdga sample_test = \n")
    # print(sample_test)

    # Now we standardize for gaussian around 0 with unit variance
    if standardize:
        std_scale = StandardScaler().fit(sample_train)
        sample_train = std_scale.transform(sample_train)
        sample_test = std_scale.transform(sample_test)

        # print("\n standardize sample_train = \n")
        # print(sample_train)
        # print("\n standardize sample_test = \n")
        # print(sample_test)

    # Now reduce number of features to number of qubits
    if pca:
        pca = PCA(n_components=n).fit(sample_train)
        sample_train = pca.transform(sample_train)
        sample_test = pca.transform(sample_test)

        # print("\n pca sample_train = \n")
        # print(sample_train)
        # print("\n pca sample_test = \n")
        # print(sample_test)

    # Scale to the range (-1,+1)
    if scale:
        samples = np.append(sample_train, sample_test, axis=0)
        minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
        sample_train = minmax_scale.transform(sample_train)
        sample_test = minmax_scale.transform(sample_test)

        # print("\n scale sample_train = \n")
        # print(sample_train)
        # print("\n scale sample_test = \n")
        # print(sample_test)

    if binarize:
        med = np.median(np.append(sample_train, sample_test, axis=0), axis=0)

        # print("\n binarize np.append(sample_train, sample_test, axis=0) = \n")
        # print(np.append(sample_train, sample_test, axis=0)[:5])
        # print("\n binarize med = \n")
        # print(med)

        transformer = Binarizer(threshold=med)
        # print("\n binarize transformer = \n")
        # print(transformer)

        sample_train = transformer.transform(sample_train)
        sample_test = transformer.transform(sample_test)

        # print("\n binarize sample_train = \n")
        # print(sample_train)
        # print("\n binarize sample_test = \n")
        # print(sample_test)

    # Pick training size number of samples from each distro
    training_input = {key: (sample_train[label_train == k, :])[:training_size]
                      for k, key in enumerate(class_labels)}
    test_input = {key: (sample_test[label_test == k, :])[:test_size]
                  for k, key in enumerate(class_labels)}

    if plot_data:
        LegitMinREBotnets = []
        LegitCharLength = []
        LegitTreeNewFeature = []
        LegitnGramReputation_Alexa = []

        DgaMinREBotnets = []
        DgaCharLength = []
        DgaTreeNewFeature = []
        DganGramReputation_Alexa = []

        # print("\n label_train = \n")
        # print(label_train)

        i = 0
        while i < len(sample_train):
            if label_train[i] == 0:
                LegitMinREBotnets.append(sample_train[i][2])
                LegitCharLength.append(sample_train[i][4])
                LegitTreeNewFeature.append(sample_train[i][5])
                LegitnGramReputation_Alexa.append(sample_train[i][6])
            else:
                DgaMinREBotnets.append(sample_train[i][2])
                DgaCharLength.append(sample_train[i][4])
                DgaTreeNewFeature.append(sample_train[i][5])
                DganGramReputation_Alexa.append(sample_train[i][6])
            i += 1

        # print("DATA \t MIN \t AVG \t MAX \n")
        # print("LegitMinREBotnets\t" + str(min(LegitMinREBotnets)) + "\t" + str(sum(LegitMinREBotnets)/len(LegitMinREBotnets)) + "\t" + str(max(LegitMinREBotnets)) + "\n")
        # print("LegitCharLength\t" + str(min(LegitCharLength)) + "\t" + str(sum(LegitCharLength)/len(LegitCharLength)) + "\t" + str(max(LegitCharLength)) + "\n")
        # print("LegitTreeNewFeature\t" + str(min(LegitTreeNewFeature)) + "\t" + str(sum(LegitTreeNewFeature)/len(LegitTreeNewFeature)) + "\t" + str(max(LegitTreeNewFeature)) + "\n")
        # print("LegitnGramReputation_Alexa\t" + str(min(LegitnGramReputation_Alexa)) + "\t" + str(sum(LegitnGramReputation_Alexa)/len(LegitnGramReputation_Alexa)) + "\t" + str(max(LegitnGramReputation_Alexa)) + "\n")
        # print("DgaMinREBotnets\t" + str(min(DgaMinREBotnets)) + "\t" + str(sum(DgaMinREBotnets)/len(DgaMinREBotnets)) + "\t" + str(max(DgaMinREBotnets)) + "\n")
        # print("DgaCharLength\t" + str(min(DgaCharLength)) + "\t" + str(sum(DgaCharLength)/len(DgaCharLength)) + "\t" + str(max(DgaCharLength)) + "\n")
        # print("DgaTreeNewFeature\t" + str(min(DgaTreeNewFeature)) + "\t" + str(sum(DgaTreeNewFeature)/len(DgaTreeNewFeature)) + "\t" + str(max(DgaTreeNewFeature)) + "\n")
        # print("DganGramReputation_Alexa\t" + str(min(DganGramReputation_Alexa)) + "\t" + str(sum(DganGramReputation_Alexa)/len(DganGramReputation_Alexa)) + "\t" + str(max(DganGramReputation_Alexa)) + "\n")

        n_bins = None
        class_labels = [r'benign', r'dga']
        colors = ['blue', 'green']
        x0 = [LegitMinREBotnets, DgaMinREBotnets]
        x1 = [LegitCharLength, DgaCharLength]
        x2 = [LegitTreeNewFeature, DgaTreeNewFeature]
        x3 = [LegitnGramReputation_Alexa, DganGramReputation_Alexa]

        # print("\n x0 MinREBotnets = \n")
        # print(LegitMinREBotnets)
        # print(DgaMinREBotnets)
        # print("\n x1 harLength = \n")
        # print(LegitCharLength)
        # print(DgaCharLength)
        # print("\n x2 TreeNewFeature = \n")
        # print(LegitTreeNewFeature)
        # print(DgaTreeNewFeature)
        # print("\n x3 nGramReputation_Alexa = \n")
        # print(LegitnGramReputation_Alexa)
        # print(DganGramReputation_Alexa)

        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
        # tmp = "Histogram of BotnedDGA Training Data: "
        # if standardize:
        #     tmp += "Standardized "
        # if pca:
        #     tmp += "PCA "
        # if scale:
        #     tmp += "Scaled "
        # plt.title(tmp)

        ax0.hist(x0, n_bins, density=True, histtype='bar', color=colors, label=class_labels)
        ax0.legend(prop={'size': 10})
        ax0.set_title('MinREBotnets')

        ax1.hist(x1, n_bins, density=True, histtype='bar', color=colors, label=class_labels)
        ax1.legend(prop={'size': 10})
        ax1.set_title('CharLength')

        ax2.hist(x2, n_bins, density=True, histtype='bar', color=colors, label=class_labels)
        ax2.legend(prop={'size': 10})
        ax2.set_title('TreeNewFeature')

        ax3.hist(x3, n_bins, density=True, histtype='bar', color=colors, label=class_labels)
        ax3.legend(prop={'size': 10})
        ax3.set_title('nGramReputation_Alexa')

        fig.tight_layout()
        plt.show()

    return sample_train, training_input, test_input, class_labels


def runTheComputation(experimentID, optimizer, feature_map, var_form, training_input, test_input):
    print(f'# Experiment  {experimentID}  = ')
    start = time.perf_counter()
    vqc = VQC(optimizer,
              feature_map,
              var_form,
              training_input,
              test_input)

    backend = BasicAer.get_backend('statevector_simulator')

    quantum_instance = QuantumInstance(backend)

    result = vqc.run(quantum_instance)

    finish = time.perf_counter()

    print('\n' + str(experimentID) + ')  Accuracy =  ' + str(result['testing_accuracy'])  + '  ; nQubits =  ' + str(feature_map.num_qubits) )
    f = open("results/result.txt", "a")
    f.write('\n' + str(experimentID) + ')  Accuracy =  ' + str(result['testing_accuracy'])  + '  ; nQubits =  ' + str(feature_map.num_qubits) )
    f.flush()
    f.close()

    return experimentID, result['testing_accuracy'], feature_map.num_qubits, round(finish - start, 2)


def main():

    start = time.perf_counter()

    # BotnetDGA data set
    plot_data = False
    training_size = 1352500
    test_size = 450833
    feature_dim = 7
    standardize = False
    pca = False
    scale = False
    binarize = False

    # f = open("results/ResultVQC_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + ".txt", "a")
    f = open("results/result.txt", "a")
    f.write(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + "   training_size = " + str(training_size) + "  test_size = " + str(test_size) + "  feature_dim = " + str(
        feature_dim) + "\n\n")
    f.flush()
    f.close()

    # random_seed = 10598
    # shots = 1024
    # seed = 1376
    # aqua_globals.random_seed = seed

    sample_train, training_input, test_input, class_labels = botnetdga(training_size=training_size,
                                                                       test_size=test_size,
                                                                       n=feature_dim,
                                                                       standardize=standardize,
                                                                       pca=pca,
                                                                       scale=scale,
                                                                       plot_data=plot_data,
                                                                       binarize=binarize)

    # print("\n sample_train = \n")
    # print(sample_train)
    # print("\n training_input = \n")
    # print(training_input)
    # print("\n test_input = \n")
    # print(test_input)
    # print("\n class_labels = \n")
    # print(class_labels)

    optimizer1 = SPSA()
    #optimizer2 = ADAM()
    #optimizer3 = AQGD()
    #optimizer4 = CG()
    #optimizer5 = COBYLA()
    #optimizer6 = L_BFGS_B()
    #optimizer7 = GSLS()
    #optimizer8 = NELDER_MEAD()
    #optimizer9 = NFT()
    #optimizer10 = P_BFGS()
    #optimizer11 = POWELL()
    #optimizer12 = SLSQP()
    #optimizer13 = TNC()

    nFeature = get_feature_dimension(training_input)

    feature_map1 = RawFeatureVector(nFeature)
    #feature_map2 = PauliFeatureMap(nFeature)
    #feature_map3 = ZFeatureMap(nFeature)
    #feature_map4 = ZZFeatureMap(nFeature)

    var_form11 = TwoLocal(feature_map1.num_qubits, ['ry', 'rz'], 'cz')
    #var_form12 = RealAmplitudes(feature_map1.num_qubits)
    #var_form13 = EfficientSU2(feature_map1.num_qubits)
    #var_form14 = ExcitationPreserving(feature_map1.num_qubits)

    #var_form21 = TwoLocal(feature_map2.num_qubits, ['ry', 'rz'], 'cz')
    #var_form22 = RealAmplitudes(feature_map2.num_qubits)
    #var_form23 = EfficientSU2(feature_map2.num_qubits)
    #var_form24 = ExcitationPreserving(feature_map2.num_qubits)

    #var_form31 = TwoLocal(feature_map3.num_qubits, ['ry', 'rz'], 'cz')
    #var_form32 = RealAmplitudes(feature_map3.num_qubits)
    #var_form33 = EfficientSU2(feature_map3.num_qubits)
    #var_form34 = ExcitationPreserving(feature_map3.num_qubits)

    #var_form41 = TwoLocal(feature_map4.num_qubits, ['ry', 'rz'], 'cz')
    #var_form42 = RealAmplitudes(feature_map4.num_qubits)
    #var_form43 = EfficientSU2(feature_map4.num_qubits)
    #var_form44 = ExcitationPreserving(feature_map4.num_qubits)


    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.submit(runTheComputation, 1, optimizer1, feature_map1, var_form11, training_input, test_input)
        #executor.submit(runTheComputation, 2, optimizer1, feature_map1, var_form12, training_input, test_input)
        #executor.submit(runTheComputation, 3, optimizer1, feature_map1, var_form13, training_input, test_input)
        #executor.submit(runTheComputation, 4, optimizer1, feature_map1, var_form14, training_input, test_input)
        #executor.submit(runTheComputation, 5, optimizer1, feature_map2, var_form21, training_input, test_input)
        #executor.submit(runTheComputation, 6, optimizer1, feature_map2, var_form22, training_input, test_input)
        #executor.submit(runTheComputation, 7, optimizer1, feature_map2, var_form23, training_input, test_input)
        #executor.submit(runTheComputation, 8, optimizer1, feature_map2, var_form24, training_input, test_input)
        #executor.submit(runTheComputation, 9, optimizer1, feature_map3, var_form31, training_input, test_input)
        #executor.submit(runTheComputation, 10, optimizer1, feature_map3, var_form32, training_input, test_input)
        #executor.submit(runTheComputation, 11, optimizer1, feature_map3, var_form33, training_input, test_input)
        #executor.submit(runTheComputation, 12, optimizer1, feature_map3, var_form34, training_input, test_input)
        #executor.submit(runTheComputation, 13, optimizer1, feature_map4, var_form41, training_input, test_input)
        #executor.submit(runTheComputation, 14, optimizer1, feature_map4, var_form42, training_input, test_input)
        #executor.submit(runTheComputation, 15, optimizer1, feature_map4, var_form43, training_input, test_input)
        #executor.submit(runTheComputation, 16, optimizer1, feature_map4, var_form44, training_input, test_input)
        #executor.submit(runTheComputation, 17, optimizer2, feature_map1, var_form11, training_input, test_input)
        #executor.submit(runTheComputation, 18, optimizer2, feature_map1, var_form12, training_input, test_input)
        #executor.submit(runTheComputation, 19, optimizer2, feature_map1, var_form13, training_input, test_input)
        #executor.submit(runTheComputation, 20, optimizer2, feature_map1, var_form14, training_input, test_input)
        #executor.submit(runTheComputation, 21, optimizer2, feature_map2, var_form21, training_input, test_input)
        #executor.submit(runTheComputation, 22, optimizer2, feature_map2, var_form22, training_input, test_input)
        #executor.submit(runTheComputation, 23, optimizer2, feature_map2, var_form23, training_input, test_input)
        #executor.submit(runTheComputation, 24, optimizer2, feature_map2, var_form24, training_input, test_input)
        #executor.submit(runTheComputation, 25, optimizer2, feature_map3, var_form31, training_input, test_input)
        #executor.submit(runTheComputation, 26, optimizer2, feature_map3, var_form32, training_input, test_input)
        #executor.submit(runTheComputation, 27, optimizer2, feature_map3, var_form33, training_input, test_input)
        #executor.submit(runTheComputation, 28, optimizer2, feature_map3, var_form34, training_input, test_input)
        #executor.submit(runTheComputation, 29, optimizer2, feature_map4, var_form41, training_input, test_input)
        #executor.submit(runTheComputation, 30, optimizer2, feature_map4, var_form42, training_input, test_input)
        #executor.submit(runTheComputation, 31, optimizer2, feature_map4, var_form43, training_input, test_input)
        #executor.submit(runTheComputation, 32, optimizer2, feature_map4, var_form44, training_input, test_input)
        #executor.submit(runTheComputation, 33, optimizer3, feature_map1, var_form11, training_input, test_input)
        #executor.submit(runTheComputation, 34, optimizer3, feature_map1, var_form12, training_input, test_input)
        #executor.submit(runTheComputation, 35, optimizer3, feature_map1, var_form13, training_input, test_input)
        #executor.submit(runTheComputation, 36, optimizer3, feature_map1, var_form14, training_input, test_input)
        #executor.submit(runTheComputation, 37, optimizer3, feature_map2, var_form21, training_input, test_input)
        #executor.submit(runTheComputation, 38, optimizer3, feature_map2, var_form22, training_input, test_input)
        #executor.submit(runTheComputation, 39, optimizer3, feature_map2, var_form23, training_input, test_input)
        #executor.submit(runTheComputation, 40, optimizer3, feature_map2, var_form24, training_input, test_input)
        #executor.submit(runTheComputation, 41, optimizer3, feature_map3, var_form31, training_input, test_input)
        #executor.submit(runTheComputation, 42, optimizer3, feature_map3, var_form32, training_input, test_input)
        #executor.submit(runTheComputation, 43, optimizer3, feature_map3, var_form33, training_input, test_input)
        #executor.submit(runTheComputation, 44, optimizer3, feature_map3, var_form34, training_input, test_input)
        #executor.submit(runTheComputation, 45, optimizer3, feature_map4, var_form41, training_input, test_input)
        #executor.submit(runTheComputation, 46, optimizer3, feature_map4, var_form42, training_input, test_input)
        #executor.submit(runTheComputation, 47, optimizer3, feature_map4, var_form43, training_input, test_input)
        #executor.submit(runTheComputation, 48, optimizer3, feature_map4, var_form44, training_input, test_input)
        #executor.submit(runTheComputation, 49, optimizer4, feature_map1, var_form11, training_input, test_input)
        #executor.submit(runTheComputation, 50, optimizer4, feature_map1, var_form12, training_input, test_input)
        #executor.submit(runTheComputation, 51, optimizer4, feature_map1, var_form13, training_input, test_input)
        #executor.submit(runTheComputation, 52, optimizer4, feature_map1, var_form14, training_input, test_input)
        #executor.submit(runTheComputation, 53, optimizer4, feature_map2, var_form21, training_input, test_input)
        #executor.submit(runTheComputation, 54, optimizer4, feature_map2, var_form22, training_input, test_input)
        #executor.submit(runTheComputation, 55, optimizer4, feature_map2, var_form23, training_input, test_input)
        #executor.submit(runTheComputation, 56, optimizer4, feature_map2, var_form24, training_input, test_input)
        #executor.submit(runTheComputation, 57, optimizer4, feature_map3, var_form31, training_input, test_input)
        #executor.submit(runTheComputation, 58, optimizer4, feature_map3, var_form32, training_input, test_input)
        #executor.submit(runTheComputation, 59, optimizer4, feature_map3, var_form33, training_input, test_input)
        #executor.submit(runTheComputation, 60, optimizer4, feature_map3, var_form34, training_input, test_input)
        #executor.submit(runTheComputation, 61, optimizer4, feature_map4, var_form41, training_input, test_input)
        #executor.submit(runTheComputation, 62, optimizer4, feature_map4, var_form42, training_input, test_input)
        #executor.submit(runTheComputation, 63, optimizer4, feature_map4, var_form43, training_input, test_input)
        #executor.submit(runTheComputation, 64, optimizer4, feature_map4, var_form44, training_input, test_input)
        #executor.submit(runTheComputation, 65, optimizer5, feature_map1, var_form11, training_input, test_input)
        #executor.submit(runTheComputation, 66, optimizer5, feature_map1, var_form12, training_input, test_input)
        #executor.submit(runTheComputation, 67, optimizer5, feature_map1, var_form13, training_input, test_input)
        #executor.submit(runTheComputation, 68, optimizer5, feature_map1, var_form14, training_input, test_input)
        #executor.submit(runTheComputation, 69, optimizer5, feature_map2, var_form21, training_input, test_input)
        #executor.submit(runTheComputation, 70, optimizer5, feature_map2, var_form22, training_input, test_input)
        #executor.submit(runTheComputation, 71, optimizer5, feature_map2, var_form23, training_input, test_input)
        #executor.submit(runTheComputation, 72, optimizer5, feature_map2, var_form24, training_input, test_input)
        #executor.submit(runTheComputation, 73, optimizer5, feature_map3, var_form31, training_input, test_input)
        #executor.submit(runTheComputation, 74, optimizer5, feature_map3, var_form32, training_input, test_input)
        #executor.submit(runTheComputation, 75, optimizer5, feature_map3, var_form33, training_input, test_input)
        #executor.submit(runTheComputation, 76, optimizer5, feature_map3, var_form34, training_input, test_input)
        #executor.submit(runTheComputation, 77, optimizer5, feature_map4, var_form41, training_input, test_input)
        #executor.submit(runTheComputation, 78, optimizer5, feature_map4, var_form42, training_input, test_input)
        #executor.submit(runTheComputation, 79, optimizer5, feature_map4, var_form43, training_input, test_input)
        #executor.submit(runTheComputation, 80, optimizer5, feature_map4, var_form44, training_input, test_input)
        #executor.submit(runTheComputation, 81, optimizer6, feature_map1, var_form11, training_input, test_input)
        #executor.submit(runTheComputation, 82, optimizer6, feature_map1, var_form12, training_input, test_input)
        #executor.submit(runTheComputation, 83, optimizer6, feature_map1, var_form13, training_input, test_input)
        #executor.submit(runTheComputation, 84, optimizer6, feature_map1, var_form14, training_input, test_input)
        #executor.submit(runTheComputation, 85, optimizer6, feature_map2, var_form21, training_input, test_input)
        #executor.submit(runTheComputation, 86, optimizer6, feature_map2, var_form22, training_input, test_input)
        #executor.submit(runTheComputation, 87, optimizer6, feature_map2, var_form23, training_input, test_input)
        #executor.submit(runTheComputation, 88, optimizer6, feature_map2, var_form24, training_input, test_input)
        #executor.submit(runTheComputation, 89, optimizer6, feature_map3, var_form31, training_input, test_input)
        #executor.submit(runTheComputation, 90, optimizer6, feature_map3, var_form32, training_input, test_input)
        #executor.submit(runTheComputation, 91, optimizer6, feature_map3, var_form33, training_input, test_input)
        #executor.submit(runTheComputation, 92, optimizer6, feature_map3, var_form34, training_input, test_input)
        #executor.submit(runTheComputation, 93, optimizer6, feature_map4, var_form41, training_input, test_input)
        #executor.submit(runTheComputation, 94, optimizer6, feature_map4, var_form42, training_input, test_input)
        #executor.submit(runTheComputation, 95, optimizer6, feature_map4, var_form43, training_input, test_input)
        #executor.submit(runTheComputation, 96, optimizer6, feature_map4, var_form44, training_input, test_input)
        #executor.submit(runTheComputation, 97, optimizer7, feature_map1, var_form11, training_input, test_input)
        #executor.submit(runTheComputation, 98, optimizer7, feature_map1, var_form12, training_input, test_input)
        #executor.submit(runTheComputation, 99, optimizer7, feature_map1, var_form13, training_input, test_input)
        #executor.submit(runTheComputation, 100, optimizer7, feature_map1, var_form14, training_input, test_input)
        #executor.submit(runTheComputation, 101, optimizer7, feature_map2, var_form21, training_input, test_input)
        #executor.submit(runTheComputation, 102, optimizer7, feature_map2, var_form22, training_input, test_input)
        #executor.submit(runTheComputation, 103, optimizer7, feature_map2, var_form23, training_input, test_input)
        #executor.submit(runTheComputation, 104, optimizer7, feature_map2, var_form24, training_input, test_input)
        #executor.submit(runTheComputation, 105, optimizer7, feature_map3, var_form31, training_input, test_input)
        #executor.submit(runTheComputation, 106, optimizer7, feature_map3, var_form32, training_input, test_input)
        #executor.submit(runTheComputation, 107, optimizer7, feature_map3, var_form33, training_input, test_input)
        #executor.submit(runTheComputation, 108, optimizer7, feature_map3, var_form34, training_input, test_input)
        #executor.submit(runTheComputation, 109, optimizer7, feature_map4, var_form41, training_input, test_input)
        #executor.submit(runTheComputation, 110, optimizer7, feature_map4, var_form42, training_input, test_input)
        #executor.submit(runTheComputation, 111, optimizer7, feature_map4, var_form43, training_input, test_input)
        #executor.submit(runTheComputation, 112, optimizer7, feature_map4, var_form44, training_input, test_input)
        #executor.submit(runTheComputation, 113, optimizer8, feature_map1, var_form11, training_input, test_input)
        #executor.submit(runTheComputation, 114, optimizer8, feature_map1, var_form12, training_input, test_input)
        #executor.submit(runTheComputation, 115, optimizer8, feature_map1, var_form13, training_input, test_input)
        #executor.submit(runTheComputation, 116, optimizer8, feature_map1, var_form14, training_input, test_input)
        #executor.submit(runTheComputation, 117, optimizer8, feature_map2, var_form21, training_input, test_input)
        #executor.submit(runTheComputation, 118, optimizer8, feature_map2, var_form22, training_input, test_input)
        #executor.submit(runTheComputation, 119, optimizer8, feature_map2, var_form23, training_input, test_input)
        #executor.submit(runTheComputation, 120, optimizer8, feature_map2, var_form24, training_input, test_input)
        #executor.submit(runTheComputation, 121, optimizer8, feature_map3, var_form31, training_input, test_input)
        #executor.submit(runTheComputation, 122, optimizer8, feature_map3, var_form32, training_input, test_input)
        #executor.submit(runTheComputation, 123, optimizer8, feature_map3, var_form33, training_input, test_input)
        #executor.submit(runTheComputation, 124, optimizer8, feature_map3, var_form34, training_input, test_input)
        #executor.submit(runTheComputation, 125, optimizer8, feature_map4, var_form41, training_input, test_input)
        #executor.submit(runTheComputation, 126, optimizer8, feature_map4, var_form42, training_input, test_input)
        #executor.submit(runTheComputation, 127, optimizer8, feature_map4, var_form43, training_input, test_input)
        #executor.submit(runTheComputation, 128, optimizer8, feature_map4, var_form44, training_input, test_input)
        #executor.submit(runTheComputation, 129, optimizer9, feature_map1, var_form11, training_input, test_input)
        #executor.submit(runTheComputation, 130, optimizer9, feature_map1, var_form12, training_input, test_input)
        #executor.submit(runTheComputation, 131, optimizer9, feature_map1, var_form13, training_input, test_input)
        #executor.submit(runTheComputation, 132, optimizer9, feature_map1, var_form14, training_input, test_input)
        #executor.submit(runTheComputation, 133, optimizer9, feature_map2, var_form21, training_input, test_input)
        #executor.submit(runTheComputation, 134, optimizer9, feature_map2, var_form22, training_input, test_input)
        #executor.submit(runTheComputation, 135, optimizer9, feature_map2, var_form23, training_input, test_input)
        #executor.submit(runTheComputation, 136, optimizer9, feature_map2, var_form24, training_input, test_input)
        #executor.submit(runTheComputation, 137, optimizer9, feature_map3, var_form31, training_input, test_input)
        #executor.submit(runTheComputation, 138, optimizer9, feature_map3, var_form32, training_input, test_input)
        #executor.submit(runTheComputation, 139, optimizer9, feature_map3, var_form33, training_input, test_input)
        #executor.submit(runTheComputation, 140, optimizer9, feature_map3, var_form34, training_input, test_input)
        #executor.submit(runTheComputation, 141, optimizer9, feature_map4, var_form41, training_input, test_input)
        #executor.submit(runTheComputation, 142, optimizer9, feature_map4, var_form42, training_input, test_input)
        #executor.submit(runTheComputation, 143, optimizer9, feature_map4, var_form43, training_input, test_input)
        #executor.submit(runTheComputation, 144, optimizer9, feature_map4, var_form44, training_input, test_input)
        #executor.submit(runTheComputation, 145, optimizer10, feature_map1, var_form11, training_input, test_input)
        #executor.submit(runTheComputation, 146, optimizer10, feature_map1, var_form12, training_input, test_input)
        #executor.submit(runTheComputation, 147, optimizer10, feature_map1, var_form13, training_input, test_input)
        #executor.submit(runTheComputation, 148, optimizer10, feature_map1, var_form14, training_input, test_input)
        #executor.submit(runTheComputation, 149, optimizer10, feature_map2, var_form21, training_input, test_input)
        #executor.submit(runTheComputation, 150, optimizer10, feature_map2, var_form22, training_input, test_input)
        #executor.submit(runTheComputation, 151, optimizer10, feature_map2, var_form23, training_input, test_input)
        #executor.submit(runTheComputation, 152, optimizer10, feature_map2, var_form24, training_input, test_input)
        #executor.submit(runTheComputation, 153, optimizer10, feature_map3, var_form31, training_input, test_input)
        #executor.submit(runTheComputation, 154, optimizer10, feature_map3, var_form32, training_input, test_input)
        #executor.submit(runTheComputation, 155, optimizer10, feature_map3, var_form33, training_input, test_input)
        #executor.submit(runTheComputation, 156, optimizer10, feature_map3, var_form34, training_input, test_input)
        #executor.submit(runTheComputation, 157, optimizer10, feature_map4, var_form41, training_input, test_input)
        #executor.submit(runTheComputation, 158, optimizer10, feature_map4, var_form42, training_input, test_input)
        #executor.submit(runTheComputation, 159, optimizer10, feature_map4, var_form43, training_input, test_input)
        #executor.submit(runTheComputation, 160, optimizer10, feature_map4, var_form44, training_input, test_input)
        #executor.submit(runTheComputation, 161, optimizer11, feature_map1, var_form11, training_input, test_input)
        #executor.submit(runTheComputation, 162, optimizer11, feature_map1, var_form12, training_input, test_input)
        #executor.submit(runTheComputation, 163, optimizer11, feature_map1, var_form13, training_input, test_input)
        #executor.submit(runTheComputation, 164, optimizer11, feature_map1, var_form14, training_input, test_input)
        #executor.submit(runTheComputation, 165, optimizer11, feature_map2, var_form21, training_input, test_input)
        #executor.submit(runTheComputation, 166, optimizer11, feature_map2, var_form22, training_input, test_input)
        #executor.submit(runTheComputation, 167, optimizer11, feature_map2, var_form23, training_input, test_input)
        #executor.submit(runTheComputation, 168, optimizer11, feature_map2, var_form24, training_input, test_input)
        #executor.submit(runTheComputation, 169, optimizer11, feature_map3, var_form31, training_input, test_input)
        #executor.submit(runTheComputation, 170, optimizer11, feature_map3, var_form32, training_input, test_input)
        #executor.submit(runTheComputation, 171, optimizer11, feature_map3, var_form33, training_input, test_input)
        #executor.submit(runTheComputation, 172, optimizer11, feature_map3, var_form34, training_input, test_input)
        #executor.submit(runTheComputation, 173, optimizer11, feature_map4, var_form41, training_input, test_input)
        #executor.submit(runTheComputation, 174, optimizer11, feature_map4, var_form42, training_input, test_input)
        #executor.submit(runTheComputation, 175, optimizer11, feature_map4, var_form43, training_input, test_input)
        #executor.submit(runTheComputation, 176, optimizer11, feature_map4, var_form44, training_input, test_input)
        #executor.submit(runTheComputation, 177, optimizer12, feature_map1, var_form11, training_input, test_input)
        #executor.submit(runTheComputation, 178, optimizer12, feature_map1, var_form12, training_input, test_input)
        #executor.submit(runTheComputation, 179, optimizer12, feature_map1, var_form13, training_input, test_input)
        #executor.submit(runTheComputation, 180, optimizer12, feature_map1, var_form14, training_input, test_input)
        #executor.submit(runTheComputation, 181, optimizer12, feature_map2, var_form21, training_input, test_input)
        #executor.submit(runTheComputation, 182, optimizer12, feature_map2, var_form22, training_input, test_input)
        #executor.submit(runTheComputation, 183, optimizer12, feature_map2, var_form23, training_input, test_input)
        #executor.submit(runTheComputation, 184, optimizer12, feature_map2, var_form24, training_input, test_input)
        #executor.submit(runTheComputation, 185, optimizer12, feature_map3, var_form31, training_input, test_input)
        #executor.submit(runTheComputation, 186, optimizer12, feature_map3, var_form32, training_input, test_input)
        #executor.submit(runTheComputation, 187, optimizer12, feature_map3, var_form33, training_input, test_input)
        #executor.submit(runTheComputation, 188, optimizer12, feature_map3, var_form34, training_input, test_input)
        #executor.submit(runTheComputation, 189, optimizer12, feature_map4, var_form41, training_input, test_input)
        #executor.submit(runTheComputation, 190, optimizer12, feature_map4, var_form42, training_input, test_input)
        #executor.submit(runTheComputation, 191, optimizer12, feature_map4, var_form43, training_input, test_input)
        #executor.submit(runTheComputation, 192, optimizer12, feature_map4, var_form44, training_input, test_input)
        #executor.submit(runTheComputation, 193, optimizer13, feature_map1, var_form11, training_input, test_input)
        #executor.submit(runTheComputation, 194, optimizer13, feature_map1, var_form12, training_input, test_input)
        #executor.submit(runTheComputation, 195, optimizer13, feature_map1, var_form13, training_input, test_input)
        #executor.submit(runTheComputation, 196, optimizer13, feature_map1, var_form14, training_input, test_input)
        #executor.submit(runTheComputation, 197, optimizer13, feature_map2, var_form21, training_input, test_input)
        #executor.submit(runTheComputation, 198, optimizer13, feature_map2, var_form22, training_input, test_input)
        #executor.submit(runTheComputation, 199, optimizer13, feature_map2, var_form23, training_input, test_input)
        #executor.submit(runTheComputation, 200, optimizer13, feature_map2, var_form24, training_input, test_input)
        #executor.submit(runTheComputation, 201, optimizer13, feature_map3, var_form31, training_input, test_input)
        #executor.submit(runTheComputation, 202, optimizer13, feature_map3, var_form32, training_input, test_input)
        #executor.submit(runTheComputation, 203, optimizer13, feature_map3, var_form33, training_input, test_input)
        #executor.submit(runTheComputation, 204, optimizer13, feature_map3, var_form34, training_input, test_input)
        #executor.submit(runTheComputation, 205, optimizer13, feature_map4, var_form41, training_input, test_input)
        #executor.submit(runTheComputation, 206, optimizer13, feature_map4, var_form42, training_input, test_input)
        #executor.submit(runTheComputation, 207, optimizer13, feature_map4, var_form43, training_input, test_input)
        #executor.submit(runTheComputation, 208, optimizer13, feature_map4, var_form44, training_input, test_input)

        # for x in EXPERIMEN:
        #     # #executor.submit(runTheComputation, x)
        # results = executor.map(runTheComputation, EXPERIMEN)


    finish = time.perf_counter()

    print(f'ALL Finished in {round( (finish-start)/3600, 2)} hour(s)')
    f = open("results/result.txt", "a")
    f.write(f'ALL Finished in {round( (finish-start)/3600, 2)} hour(s)')
    f.write(f'ALL Finished in {round( (finish-start), 2)} second(s)')
    f.flush()
    f.close()


if __name__ == '__main__':
    main()