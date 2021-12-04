import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from matplotlib.colors import ListedColormap
import os
import logging

plt.style.use("fivethirtyeight")

def prepare_data(df):
    """
    It is used to separate the data into features and target.

    Args:
    df (pd.DataFrame): It is the pandas dataframe that contains the data.

    Returns:
        tuple: It returns the tuples of dependent and independent variables.
    """
    logging.info("Preparing data")
    X = df.drop("y", axis = 1)
    y = df["y"]
    return X, y

def save_model(model, filename):
    """It saves the trained model in the specified directory.

    Args:
        model (python object): trained model.
        filename (str): Path to save the trained model.
    """
    logging.info("Saving the trained model")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    filePath = os.path.join(model_dir, filename)
    joblib.dump(model, filePath)
    logging.info(f"Model saved in {filePath}")

def save_plot(df, filename, model):
    """
    :param df: It is the dataframe 
    :param filename: It is the name of the file to save the plot
    :param model: It is the trained model
    """
    def _create_base_plot(df):
        logging.info("Creating the base plot")
        df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
        figure = plt.gcf() # get current figure
        figure.set_size_inches(10, 8)

    def _plot_decision_regions(X, y, classifier, resolution=0.02):
        logging.info("Plotting decision regions")
        colors = ("red",  "blue", "lightgreen", "gray", "cyan")
        cmap = ListedColormap(colors[: len(np.unique(y))])
        X = X.values # as an array
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                               np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        plt.plot()


    X, y = prepare_data(df)
    _create_base_plot(df)
    _plot_decision_regions(X, y, model)
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    plotPath = os.path.join(plot_dir, filename)
    plt.savefig(plotPath)
    logging.info(f"Saving the plot at {plotPath}")