"""
Feature analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve


class FeatureAnalysis:
    """
    Class to analyse the features from titanic challenge.
    Call ... to work on input features, then call
    process_test_dataset to work on test data.
    get_correlation_numericalvalues will create graphs that show the correlation
     among features.
    """
    #remake definition at the end

    #def __init__(self):


    def get_correlation_numericalvalues(self, dataframe, numerical_tag) -> None:
        """
        Receives the dataframe and creates the graphs to analyse correlation
        among numerical features.

        :dataframe: pd.DataFrame
        :numerical_tag: indexes to be considered
        :return: None
        """
        sns.set(style='white', context='notebook', palette='deep')

        # Correlation matrix between numerical values (SibSp Parch Age and Fare
        # values) and Survived
        g = sns.heatmap(
            dataframe[numerical_tag].corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm")
        plt.show()

        # Explore SibSp feature vs Survived
        self.explore_feature_output(dataframe, "SibSp", "Survived")
        plt.show()

        # Explore Parch feature vs Survived
        self.explore_feature_output(dataframe, "Parch", "Survived")
        plt.show()

        # Explore Age vs Survived
        g = sns.FacetGrid(dataframe, col='Survived')
        g = g.map(sns.distplot, "Age")
        plt.show()

        # Explore Age distibution
        g = sns.kdeplot(
            dataframe["Age"][
                (dataframe["Survived"] == 0) & (dataframe["Age"].notnull())],
            color="Red",
            shade=True)
        g = sns.kdeplot(
            dataframe["Age"][
                (dataframe["Survived"] == 1) & (dataframe["Age"].notnull())],
            ax=g,
            color="Blue",
            shade=True)
        g.set_xlabel("Age")
        g.set_ylabel("Frequency")
        g = g.legend(["Not Survived", "Survived"])
        plt.show()

        # Explore Fare distribution
        g = sns.distplot(
            dataframe["Fare"],
            color="m",
            label="Skewness : %.2f" % (dataframe["Fare"].skew()))
        g = g.legend(loc="best")
        plt.show()

    def analyse_categoricalvalues(self, dataframe) -> None:
        """
        Receives the dataframe and creates the graphs to analyse the categorical
        features.

        :dataframe: pd.DataFrame
        :return: None
        """
        sns.set(style='white', context='notebook', palette='deep')

        # Explore Sex vc Survived
        g = sns.barplot(x="Sex", y="Survived", data=dataframe)
        g = g.set_ylabel("Survival Probability")
        plt.show()

        # Explore Pclass vs Survived
        self.explore_feature_output(dataframe, "Pclass", "Survived")
        plt.show()

        # Explore Pclass vs Survived by Sex
        g = sns.catplot(
            x="Pclass",
            y="Survived",
            hue="Sex",
            data=dataframe,
            height=6,
            kind="bar",
            palette="muted")
        g.despine(left=True)
        g = g.set_ylabels("survival probability")
        plt.show()

        # Explore Embarked vs Survived
        self.explore_feature_output(dataframe, "Embarked", "Survived")
        plt.show()

        # Explore Pclass vs Embarked
        g = sns.catplot(
            "Pclass",
            col="Embarked",
            data=dataframe,
            height=6,
            kind="count",
            palette="muted")
        g.despine(left=True)
        g = g.set_ylabels("Count")
        plt.show()

    def explore_feature_output(self, dataframe, feature, output) -> None:
        """
        Creates graph to explore the influence of one feature on the output or
        the correlation among two features.

        :dataframe: pd.DataFrame
        :feature: input feature
        :output: output
        :return: None
        """
        g = sns.catplot(
            x=feature,
            y=output,
            data=dataframe,
            kind="bar",
            height=6,
            palette="muted")
        g.despine(left=True)
        g = g.set_ylabels("Survival Probability")
        plt.show()
