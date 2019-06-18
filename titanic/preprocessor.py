"""
Class for preprocessing titanic test.
"""

from typing import List
from typing import Tuple

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

IGNORE_COLUMNS = ['PassengerId', 'Name', 'Cabin', 'Ticket']


class Preprocessor:
    """
    Class for preprocessing data from titanic challenge.
    Call process_training_dataset to work on training data, then call
    process_test_dataset to work on test data.
    get_train_datasets will return a tuple with input,output lists.
    get_test_dataset will return a list with inputs for prediction.
    """
    def __init__(self):
        self.age_mean = 0
        self.train_input = []
        self.train_output = []
        self.test_input = []
        self.encoders = {}
        self.params = {}
        self.normalizer = None
        self.thingies = []
        self.train_input_value = None
        self.test_input_value = None

    @staticmethod
    def map_to_divisions(
            data_array: List[float],
            number_divisions: int = None,
            divisions: List[float] = None):     # (...) -> np.array;
        """
        Receives a list with data and will divide its elements into a given
        number of classes, e.g. an array with numbers from 1 to 100 with 10
        divisions will have 10 elements in class 1, 10 in class 2, and so on.
        If the divisions were given, they will be used instead.
        Divisions are expected as list of values, with the highest value less
        or equal to highest value of data array, each value representing the
        upper limit of a class. With this, to achieve same result of previous
        example, divisions [10, 20, 30, 40, ..., 100] should be given.

        Returns list with classified values. Classes are integer.

        Either one of number_divisions or divisions parameters must be given,
        or AttributeError is risen. Same error is risen for both parameters
        given.

        :data_array: list[float]
        :number_divisions: int
        :divisions: list[float]

        :return: np.array
        """

        if number_divisions is None and divisions is None:
            raise AttributeError(
                "No classifying strategy given when one is expected.")
        if number_divisions is not None and divisions is not None:
            raise AttributeError(
                "Both parameters given when only one is expected.")

        strategy = 'number'     # We are using default divisions
        if divisions is not None:
            strategy = 'list'   # We are using given divisions

        input_array = np.array(data_array)
        array_max = np.amax(input_array)

        # The strategy will tell how to build divisions array with (limit, ind)
        if strategy == 'list':
            divisions.sort()
            if np.amax(np.array(divisions)) < array_max:
                divisions.append(array_max)
            for ind in range(len(divisions)):
                divisions[ind] = (divisions[ind], ind+1)
        elif strategy == 'number':
            divisions = []
            array_min = np.amin(input_array)
            division_step = (array_max - array_min) / number_divisions
            number_steps = 0
            while number_steps <= number_divisions:
                number_steps += 1
                divisions.append((division_step*number_steps, number_steps))
        for ind in range(len(input_array)):
            for limit, class_ in divisions:
                if input_array[ind] <= limit:
                    input_array[ind] = class_
                    break
        return input_array

    @staticmethod
    def remove_outliers(dataframe, features, boundary_factor: float = 1.5):
        """
        Receives a dataframe and features to have outliers removed.
        Implements IQR method with factor to get outer boundaries set by
        boundary_factor.

        :dataframe: pd.DataFrame
        :features: List[str]
        :boundary_factor: float

        :return: pd.DataFrame
        """
        for col in features:
            # Calculates quartiles, IQR and boundaries
            first_quartile = np.percentile(dataframe[col], 25)
            third_quartile = np.percentile(dataframe[col], 75)
            iqr = third_quartile - first_quartile
            min_bound = first_quartile - boundary_factor * iqr
            max_bound = third_quartile + boundary_factor * iqr

            # Based on boundaries, get indexes to remove and remove them
            outliers = []
            for index in range(len(dataframe[col])):
                if not (min_bound < dataframe[col][index] < max_bound):
                    outliers.append(index)
            dataframe = dataframe.drop(labels=outliers, axis=0)

        return dataframe

    def get_train_datasets(self) -> Tuple[List[int], List[int]]:
        """
        Returns a tuple with input, output datasets from training.

        :return: Tuple(List(int), List(int))
        """
        return (self.train_input_value, self.train_output)

    def get_test_dataset(self) -> List[int]:
        """
        Returns the test dataset for prediction.

        :return: List(int)
        """
        return self.test_input_value

    def process_training_dataset(self, file: str) -> None:
        """
        Receives the path to a file and gets the parametes from it.

        :file: str
        :return: None
        """

        # Import the dataset
        dataframe = pd.read_csv(file)
        # Removed cabin and name columns
        filtered_dataframe = dataframe.drop(
            IGNORE_COLUMNS, axis=1)
        output = filtered_dataframe.iloc[:, 0].values
        input_value = filtered_dataframe.iloc[:, 1:9].values

        # Finding missing values
        print('Missing values:\n')
        print(filtered_dataframe.isnull().sum())

        # Gets age mean and saves as parameter
        self.params['age_mean'] = filtered_dataframe['Age'].mean()
        self.params['fare_mean'] = filtered_dataframe['Fare'].mean()

        input_value[:, 2] = filtered_dataframe['Age'].fillna(
            self.params['age_mean'])
        input_value[:, 5] = filtered_dataframe['Fare'].fillna(
            self.params['fare_mean'])

        # Breakes age into 3 classes: <18, 18~60 and >60.
        input_value[:, 2] = self.map_to_divisions(
            input_value[:, 2],
            divisions=[18, 60])

        # Sex converted to 0/1
        self.encoders['sex'] = LabelEncoder()
        input_value[:, 1] = self.encoders['sex'].fit_transform(
            input_value[:, 1])

        # Embarked converted to labels
        self.encoders['embarked'] = LabelEncoder()
        input_value[:, 6] = self.encoders['embarked'].fit_transform(
            input_value[:, 6].astype(str))

        # Encoding P Class, Sib, Parch and Embarked
        self.encoders['onehotencoder'] = OneHotEncoder(
            categorical_features=[0, 2, 3, 4, 6],
            handle_unknown='ignore')
        input_value = self.encoders['onehotencoder'].fit_transform(
            input_value).toarray()

        # Normalization for 2 numeric features -> -2
        self.normalizer = MinMaxScaler(feature_range=(0, 1))
        input_value[:, -2:] = self.normalizer.fit_transform(
            input_value[:, -2:])

        print("First line of end matrix:")
        print(input_value[0][:])

        self.train_input_value = input_value
        self.train_output = output

    def process_test_dataset(self, file: str) -> None:
        """
        Receives the path to a file and gets the works its data for processing.

        :file: str
        :return: None
        """
        dataframe = pd.read_csv(file)
        # Removed cabin and name columns
        filtered_dataframe = dataframe.drop(
            columns=IGNORE_COLUMNS)

        input_value = filtered_dataframe.iloc[:, :].values

        # Finding missing values
        print('Missing values:\n')
        print(filtered_dataframe.isnull().sum())

        input_value[:, 2] = filtered_dataframe['Age'].fillna(
            self.params['age_mean'])
        input_value[:, 5] = filtered_dataframe['Fare'].fillna(
            self.params['fare_mean'])

        input_value[:, 2] = self.map_to_divisions(
            input_value[:, 2],
            divisions=[18, 60])

        # Sex converted to 0/1
        input_value[:, 1] = self.encoders['sex'].transform(input_value[:, 1])

        # Embarked converted to labels
        input_value[:, 6] = self.encoders['embarked'].transform(
            input_value[:, 6])

        # Encoding P Class, Sib, Parch and Embarked
        input_value = self.encoders['onehotencoder'].transform(
            input_value).toarray()

        # Normalization for 2 numeric features -> -2
        input_value[:, -2:] = self.normalizer.transform(input_value[:, -2:])

        print("First line of end matrix:")
        print(input_value[0][:])

        self.test_input_value = input_value


if __name__ == '__main__':
    a = pd.DataFrame([1, 2, 3, 4, 5, 6, 7])
    d = pd.DataFrame(
        [[1, 4], [2, 8], [3, 12], [4, 20], [5, 500], [6, 17], [100, 23]])

    print(Preprocessor.remove_outliers(a, [0]))
    print(Preprocessor.remove_outliers(d, [0, 1]))
