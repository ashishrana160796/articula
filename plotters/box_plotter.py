import matplotlib.pyplot as plt
import numpy as np

from common.data_utils import load_informaticup_text_data
from models.intrinsic_dim_estimator import IntrinsicDimensionEstimator
from spoofer.baseline_text_spoofer import BaselineTextSpoofer    

def get_box_plots(data, labels, x_label, y_label, title, colors, notch=True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))


    if notch:
        # notch shape box plot
        bplot = ax.boxplot(data,
                            notch=True,  # notch shape
                            vert=True,  # vertical box alignment
                            patch_artist=True,  # fill with color
                            labels=labels)  # will be used to label x-ticks
    else:
        # rectangular box plot
        bplot = ax.boxplot(data,
                            vert=True,  # vertical box alignment
                            patch_artist=True,  # fill with color
                            labels=labels)  # will be used to label x-ticks
    
    ax.set_title(title)

    # fill with colors
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    # adding horizontal grid lines
    ax.yaxis.grid(True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.show()


if __name__ == '__main__':
    # # Random test data for testing the plot functionality
    # np.random.seed(19680801)
    # data = [np.random.normal(0, std, size=100) for std in range(1, 4)]
    # labels = ['x1', 'x2', 'x3']
    # title = 'Notched box plot'
    # colors = ['pink', 'lightblue', 'lightgreen']
    # x_label = 'Three separate samples'
    # y_label = 'Observed values'
    # get_box_plots(data, labels, x_label, y_label, title, colors)

    # Obtaining intrinsic dimensionality of original, baseline spoofed, and additonally spoofed datasets
    labels = ['Original', 'Spoofed', 'Added Data']
    colors = ['pink', 'lightblue', 'lightgreen']
    x_label = 'Intrinsic Dimensionality'
    y_label = 'Comparitive Approaches'
    title = 'Intrinsic Dimensionality Comparitive Analysis'
    dim_box_plt_data = [
                [10.175535785921744, 7.703852131331417, 12.739321552009397, 8.365127448156105, 6.795671178496798, 2.145307259338947, 10.778247873329184, 9.548013188826413, 10.035087908005172, 12.871777291967913, 12.332223352700904, 9.768352981990587, 9.413370714634416],
                [7.240186275511038, 8.097683091344889, 11.561927231954726, 8.499130774304296, 6.990383153868127, 2.645309961935223, 6.1029185731668845, 9.298068931780703, 10.621414695813167, 11.676305610323594, 11.961312559963996, 9.762547630898549, 5.882497868640205],
                [10.330198209185772, 9.728221295687813, 11.934292948249551, 8.702343716554733, 9.66787686655942, 2.6979414553344303, 8.227123370190643, 10.195097501073459, 10.207375317305047, 11.924321723647902, 12.176436781228045, 11.14130389875645, 8.532342921076491]
            ]
    informaticup_data = load_informaticup_text_data()
    dim_box_plt_data = []
    int_dim_estimator = IntrinsicDimensionEstimator(informaticup_data)
    org_intrinsic_dim_list = list(int_dim_estimator.get_mle().ravel())
    dim_box_plt_data.append(org_intrinsic_dim_list)
    del int_dim_estimator
    text_spoofer = BaselineTextSpoofer(informaticup_data)
    spoof_intrinsic_dim_tuple_list = list(text_spoofer.spoof_text())
    spoof_intrinsic_dim_list = [x[1] for x in spoof_intrinsic_dim_tuple_list]
    dim_box_plt_data.append(spoof_intrinsic_dim_list)
    del text_spoofer
    text_spoofer = BaselineTextSpoofer(informaticup_data, add_info_mutation=True)
    add_text_intrinsic_dim_tuple_list = list(text_spoofer.spoof_text())
    add_text_intrinsic_dim_list = [x[1] for x in add_text_intrinsic_dim_tuple_list]
    dim_box_plt_data.append(add_text_intrinsic_dim_list)
    del text_spoofer
    get_box_plots(dim_box_plt_data, labels, x_label, y_label, title, colors)
