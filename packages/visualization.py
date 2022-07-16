#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plotting functions for the model.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def kdeplot(data, x, hue):
    """
    Plot KDE of data grouped by hue
    """
    plt.figure(figsize=(15, 5))
    sns.kdeplot(data=data, x=x, hue=hue, shade=True)
    plt.title(f'Distribution of {x} grouped by {hue}')
    plt.xlabel(x)
    plt.ylabel(None)
    plt.yticks([])
    plt.show()


def plot_loss(nn_metrics_data):
    """
    Plots the loss for the training and validation datasets.
    """
    plt.plot(nn_metrics_data['loss'], 'o-', label='training loss')
    plt.plot(nn_metrics_data['val_loss'], 'o-', label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


def plot_acc(nn_metrics_data):
    """
    Plots the accuracy for the training and validation datasets.
    """
    plt.plot(nn_metrics_data['accuracy'], 'o-', label='training accuracy')
    plt.plot(nn_metrics_data['val_accuracy'], 'o-', label='validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
