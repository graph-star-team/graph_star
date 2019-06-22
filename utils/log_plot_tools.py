#! /usr/bin/env python3
# -*- coding=UTF-8  -*-
import sys
import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import argparse

global counter
counter = 1

global dataset_name
dataset_name = ""

name_dict =	{
  "1": "Train Loss for Graph Classification",
  "2": "Train Accuracy for Graph Classification",
  "3": "Test Loss for Graph Classification",
  "4": "Test Accuracy for Graph Classification"
}


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def getData(rawDataPath):
    '''重组log中的数据 '''
    train_loss_array = []
    train_acc_array = []
    test_loss_array = []
    test_acc_array = []

    epoch_list = []

    global  max_epoch
    global  dataset_name

    if not os.path.exists(rawDataPath):
        return -1
    with open(rawDataPath, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.replace('[1;32m', '')
            line = line.replace('[0m', '')
            line = line.replace(' ', '')
            if line == '\n':
                continue
            #elif not line.startswith("NCI"): #("PROTEINSEpoch:"):
            elif "TRAINLoss" not in line:
                continue
            else:
                items = line.strip('\n').split(',')
                epoch_id = items[0].split(':')
                train_loss = items[1].split(':')
                train_acc = items[2].split(':')
                test_loss = items[5].split(':')
                test_acc = items[6].split(':')

                dataset_name = epoch_id[0].replace('Epoch', '')
                train_loss_array.append(float(train_loss[1]))
                train_acc_array.append(float(train_acc[1]))
                test_loss_array.append(float(test_loss[1]))
                test_acc_array.append(float(test_acc[1]))
                epoch_list.append(int(epoch_id[1]))

    max_epoch = np.max(epoch_list)
    train_loss_all = np.array_split(train_loss_array, 10)
    train_acc_all = np.array_split(train_acc_array, 10)
    test_loss_all = np.array_split(test_loss_array, 10)
    test_acc_all = np.array_split(test_acc_array, 10)

    print("Data separated....")
    train_loss_all = np.vstack(train_loss_all)
    train_acc_all = np.vstack(train_acc_all)
    test_loss_all = np.vstack(test_loss_all)
    test_acc_all = np.vstack(test_acc_all)

    return max_epoch, train_loss_all, train_acc_all, test_loss_all, test_acc_all


def plot_fig(data_frame, cut_point):
    data = data_frame #copy.deepcopy(data_frame)
    # Make a data frame
    global counter
    global dataset_name
    plt.figure(counter)
    plot_title = name_dict.get(str(counter))
    counter += 1

    df = pd.DataFrame({'x': range(0, cut_point),
                       'y0': data[0, :cut_point],  'y1': data[1, :cut_point],
                       'y2': data[2, :cut_point],  'y3': data[3, :cut_point],
                       'y4': data[4, :cut_point],  'y5': data[5, :cut_point],
                       'y6': data[6, :cut_point],  'y7': data[7, :cut_point],
                       'y8': data[8, :cut_point],  'y9': data[9, :cut_point]})

    # style
    plt.style.use('seaborn-darkgrid')

    # create a color palette
    palette = plt.get_cmap('Set1')

    # multiple line plot
    num = 0
    for column in df.drop('x', axis=1):
        num += 1
        plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=0.4, alpha= 1, label=column)

    # Add legend
    plt.legend(loc=2, ncol=2)

    # Add titles
    plt.title(str(dataset_name.capitalize()) + " " + plot_title, fontsize=10, loc='center',  color='black')  # fontsize=12, fontweight=0,
    plt.xlabel("Epoch")
    plt.ylabel("Score")

def main(_args):

    parser = argparse.ArgumentParser(description='log_plot_tools')
    parser.add_argument('--show_plots', type=str2bool, default=True)
    parser.add_argument('--save_as_pdf', type=str2bool, default=False)
    parser.add_argument('--log_dir', type=str, default='./05-21_17-27-proteins_3layer.txt')

    args = parser.parse_args(_args)

    max_epoch ,train_loss_all, train_acc_all, test_loss_all, test_acc_all = getData(str(args.log_dir))
    cut_point = max_epoch

    plot_fig(train_loss_all, cut_point)
    plot_fig(train_acc_all, cut_point)
    plot_fig(test_loss_all, cut_point)
    plot_fig(test_acc_all, cut_point)

    if args.save_as_pdf:
        pdf = matplotlib.backends.backend_pdf.PdfPages(str(dataset_name) + "_save_all_plots.pdf")
        for fig in range(1, plt.gcf().number + 1):
            pdf.savefig(fig, bbox_inches='tight', dpi = 600)
        pdf.close()


    if args.show_plots:
        plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])