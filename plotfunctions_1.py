#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:14:07 2019

@author: michaelboles
"""

# import packages
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# plot histogram

def plot_hist(data, binwidth, textbox, props, text_pos, xmin, xmax, ymin, ymax, yint, xlabel, ylabel, save, figure_name):
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    bins = np.arange(round(min(data),1), max(data) + binwidth, binwidth)
    props = dict(facecolor='white', alpha=1.0)

    ax.hist(data, bins, edgecolor = 'black', facecolor = 'blue')
    
    plt.xlim(xmin, xmax); plt.xlabel(xlabel, fontsize = 18, fontname = 'Helvetica')
    plt.ylim(ymin, ymax)
    plt.ylabel(ylabel, fontsize = 18)
    ax.tick_params(axis = 'x', labelsize = 14); ax.tick_params(axis = 'y', labelsize = 14)
    
    for tick in ax.get_xticklabels():
        tick.set_fontname('Helvetica')
    for tick in ax.get_yticklabels():
        tick.set_fontname('Helvetica')
        
    ax.text(text_pos, 0.95, textbox, transform = ax.transAxes, fontsize = 18, 
            fontname = 'Helvetica', verticalalignment = 'top', bbox = props)

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.unicode_minus'] = False
    plt.grid(); ax.grid(color=(.9, .9, .9)); ax.set_axisbelow(True)

    fig.tight_layout()
    
    import matplotlib.ticker as ticker
    xscale = 1
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/xscale))
    ax.xaxis.set_major_formatter(ticks)
    
    # force integers on y-axis
    if yint == True:
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if save == True:
        plt.savefig(figure_name, dpi = 600)
    plt.show()
    
def plot_hist_log_y(data, binwidth, textbox, props, text_pos, xmin, xmax, ymin, ymax, xlabel, ylabel, yticks, save, figure_name):
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    bins = np.arange(round(min(data),1), max(data) + binwidth, binwidth)
    props = dict(facecolor='white', alpha=1.0)

    ax.hist(data, bins, edgecolor = 'black', facecolor = 'blue')

    plt.xlim(xmin, xmax); plt.xlabel(xlabel, fontsize = 18, fontname = 'Helvetica')

    plt.yscale('log')
    plt.yticks(yticks)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.ylabel(ylabel, fontsize = 18)
    ax.tick_params(axis = 'x', labelsize = 14); ax.tick_params(axis = 'y', labelsize = 14)

    plt.ylim(ymin, ymax)

    for tick in ax.get_xticklabels():
        tick.set_fontname('Helvetica')
    for tick in ax.get_yticklabels():
        tick.set_fontname('Helvetica')

    ax.text(text_pos, 0.95, textbox, transform = ax.transAxes, fontsize = 18, 
            fontname = 'Helvetica', verticalalignment = 'top', bbox = props)

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.unicode_minus'] = False
    plt.grid(); ax.grid(color=(.9, .9, .9)); ax.set_axisbelow(True)

    fig.tight_layout()

    import matplotlib.ticker as ticker
    xscale = 1
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/xscale))
    ax.xaxis.set_major_formatter(ticks)

    if save == True:
        plt.savefig(figure_name, dpi = 600)
    plt.show()