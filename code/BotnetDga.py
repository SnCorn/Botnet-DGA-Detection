import matplotlib.pyplot as plt
import numpy as np
from qiskit import BasicAer
from qiskit.ml.datasets import ad_hoc_data, breast_cancer
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua.algorithms import SklearnSVM, QSVM
from qiskit.circuit.library import ZZFeatureMap
import datetime
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import joblib
import time



def load_data(data_file_name):

    with open(data_file_name) as csv_file:
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


def load_botnet_dga_dataset():

    data, target, target_names = load_data('../data/BotnetDgaDataset.csv')
    # columns = Entropy,REAlexa,MinREBotnets,InformationRadius,CharLength,TreeNewFeature,nGramReputation_Alexa,Class

    return data, target


def botnet_dga_dataset(training_size, test_size, n, plot_data=False):

    """ returns botnet dga dataset """
    class_labels = [r'A', r'B']
    data, target = load_botnet_dga_dataset()
    sample_train, sample_test, label_train, label_test = \
        train_test_split(data, target, test_size=0.3, random_state=12)

    # Now we standardize for gaussian around 0 with unit variance
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)

    # Now reduce number of features to number of qubits
    pca = PCA(n_components=n).fit(sample_train)
    sample_train = pca.transform(sample_train)
    sample_test = pca.transform(sample_test)

    # Scale to the range (-1,+1)
    samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    sample_train = minmax_scale.transform(sample_train)
    sample_test = minmax_scale.transform(sample_test)

    # Pick training size number of samples from each distro
    training_input = {key: (sample_train[label_train == k, :])[:training_size]
                      for k, key in enumerate(class_labels)}
    test_input = {key: (sample_test[label_test == k, :])[:test_size]
                  for k, key in enumerate(class_labels)}

    if plot_data:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise NameError('Matplotlib not installed. Please install it before plotting')
        for k in range(0, 2):
            plt.scatter(sample_train[label_train == k, 0][:training_size],
                        sample_train[label_train == k, 1][:training_size])

        plt.title("PCA dimension reduced Botnet DGA training dataset")
        plt.show()

    return sample_train, training_input, test_input, class_labels



filenameClfQSVM = 'clfQSVM.sav'
serializeClassifier = True
deSerializeClassifier = True


feature_dim = 2  # dimension of each data point
training_dataset_size = 20
testing_dataset_size = 10
random_seed = 10598
shots = 1024


if(serializeClassifier):

    print ("\n\n START serializeClassifier) " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start_time = time.time()

    sample_Total, training_input, test_input, class_labels = botnet_dga_dataset(
        training_size=training_dataset_size,
        test_size=testing_dataset_size,
        n=feature_dim, plot_data=True
    )
    print ("\n\n 1) " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if (deSerializeClassifier):
        datapoints, class_to_label = split_dataset_to_data_and_labels(test_input)
        print(class_to_label)
        print ("\n\n 2) " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    feature_map = ZZFeatureMap(feature_dim, reps=2)
    print ("\n\n 3) " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    svm = QSVM(feature_map, training_input, test_input, None)# the data for prediction can be fed later.
    svm.random_seed = random_seed
    print ("\n\n 4) " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print('save the model to disk')
    joblib.dump(svm, filenameClfQSVM)
    print(filenameClfQSVM)

    # print("detail tentang Training dataset: ", training_input)
    # print("detail tentang Testing dataset: ", test_input)

    print ("\n\n FINISH serializeClassifier) " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("--- %s seconds ---" % (time.time() - start_time))



if(deSerializeClassifier):

    print ("\n\n START deSerializeClassifier) " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start_time = time.time()

    print('load the model from disk')
    svm = joblib.load(filenameClfQSVM)

    backend = BasicAer.get_backend('qasm_simulator')
    print ("\n\n 5) " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    quantum_instance = QuantumInstance(backend, shots=shots, seed_simulator=random_seed, seed_transpiler=random_seed)
    print ("\n\n 6) " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    result = svm.run(quantum_instance)
    print ("\n\n 7) " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print("kernel matrix during the training:")
    kernel_matrix = result['kernel_matrix_training']
    img = plt.imshow(np.asmatrix(kernel_matrix),interpolation='nearest',origin='upper',cmap='bone_r')
    plt.show()

    print("testing success ratio: ", result['testing_accuracy'])

    # print("detail result: ", result)
    print ("\n\n 8) " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if (serializeClassifier):
        predicted_labels = svm.predict(datapoints[0])
        print ("\n\n 9) " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        predicted_classes = map_label_to_class_name(predicted_labels, svm.label_to_class)
        print("ground truth: {}".format(datapoints[1]))
        print("preduction:   {}".format(predicted_labels))

    print ("\n\n FINISH deSerializeClassifier) " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("--- %s seconds ---" % (time.time() - start_time))


