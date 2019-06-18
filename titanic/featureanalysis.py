"""
Feature analysis.
"""

import seaborn as sns
import matplotlib.pyplot as plt


class FeatureAnalysis:
    """
    Class to analyse the features from titanic challenge.
    Call ... to work on input features, then call
    process_test_dataset to work on test data.
    get_correlation_numerical_values will create graphs that show the
    correlation among features.
    """

    def __init__(self, dataframe, show_plots=True):
        """
        Receives a dataframe to be worked.

        :dataframe: pd.DataFrame
        :show_plots: boolean : (optional) allows user to choose whether to plot
                               diagrams or not.
        """
        self.dataframe = dataframe
        self.show_plots = show_plots
        sns.set(style='white', context='notebook', palette='deep')

    def get_correlation_numerical_values(self, indexes) -> None:
        """
        Creates the graphs to analyse correlation among numerical features.
        First in list indexes will be the one to have correlation checked
        against others (you maybe want it to be the output)

        :indexes: List[str] : dataframe indexes to be considered
        :return: None
        """
        sns.heatmap(
            self.dataframe[indexes].corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm")
        if self.show_plots:
            plt.show()

    def distribution_plot_against_categorical(self, index_one, index_two):
        """
        Plots index_two against caregories of index_one, e.g. distribution of
        index_two for index_one == 0 and dist of index_two for index_one == 1.
        index_one should be categorical and 1 should be numerical.

        :index_one: str : "Categorical" feature
        :index_two: str : Numerical feature
        """
        # Explore Age vs Survived
        tmp = sns.FacetGrid(self.dataframe, col=index_one)
        tmp.map(sns.distplot, index_two)
        if self.show_plots:
            plt.show()

    def plot_against_true_false_var(self, index, tf_index):
        """
        Creates a graph to plot one feature against a boolean feature.
        Boolean is matched using "dataframe[tf_index] == [0 (or) 1]".

        FIXME: maybe use bool(feature) / not bool(feature) to evaluate instead

        :index: str : index of dataframe to be plotted against True/F var
        :tf_index: str : index of dataframe True/False variable for separation
        :return: None
        """
        # Explore Age distibution
        tmp = sns.kdeplot(
            self.dataframe[index][
                (self.dataframe[tf_index] == 0)
                & (self.dataframe[index].notnull())],
            color="Red",
            shade=True)
        tmp = sns.kdeplot(
            self.dataframe[index][
                (self.dataframe[tf_index] == 1)
                & (self.dataframe[index].notnull())],
            ax=tmp,
            color="Blue",
            shade=True)
        tmp.set_xlabel(index)
        tmp.set_ylabel("Frequency")
        tmp.legend(["Not " + tf_index, index])
        if self.show_plots:
            plt.show()

    def explore_distribution_plot(self, index):
        """
        Calls distribution plot for data skewness.
        :index: str : index of dataframe to be plotted.
        """
        tmp = sns.distplot(
            self.dataframe[index],
            color="m",
            label="Skewness : %.2f" % (self.dataframe[index].skew()))
        tmp.legend(loc="best")
        if self.show_plots:
            plt.show()

    def categorical_plot_frequency(self, index_one, index_two):
        """
        Plots count of index_one vs index_two.
        :index_one: str : Variable to be plotted against index_two
        :index_two: str : Variable to be plotted against index_one
        """
        tmp = sns.barplot(x=index_one, y=index_two, data=self.dataframe)
        tmp.set_ylabel("Probability of " + index_two)
        if self.show_plots:
            plt.show()

    def categorical_plot_three_vars(self, index_one, index_two, index_three):
        """
        Plots count of index_one vs index_two by (hue) index_three.
        :index_one: str : Feature to be plotted
        :index_two: str : Feature to be plotted
        :index_three: str : Feature to be plotted as hue
        """
        tmp = sns.catplot(
            x=index_one,
            y=index_two,
            hue=index_three,
            data=self.dataframe,
            height=6,
            kind="bar",
            palette="muted")
        tmp.despine(left=True)
        tmp.set_ylabels("probability of " + index_two)
        if self.show_plots:
            plt.show()

    def categorical_plot_two_vars(self, index_one, index_two):
        """
        Plots count of index_one vs index_two.
        :index_one: str : Variable to be plotted against index_two
        :index_two: str : Variable to be plotted against index_one
        """
        tmp = sns.catplot(
            index_one,
            col=index_two,
            data=self.dataframe,
            height=6,
            kind="count",
            palette="muted")
        tmp.despine(left=True)
        tmp.set_ylabels("Count")
        if self.show_plots:
            plt.show()

    def explore_feature_output(self, feature, output) -> None:
        """
        Creates graph to explore the influence of one feature on the output or
        the correlation among two features.

        :feature: input feature
        :output: output
        :return: None
        """
        tmp = sns.catplot(
            x=feature,
            y=output,
            data=self.dataframe,
            kind="bar",
            height=6,
            palette="muted")
        tmp.despine(left=True)
        tmp.set_ylabels("Survival Probability")
        if self.show_plots:
            plt.show()
