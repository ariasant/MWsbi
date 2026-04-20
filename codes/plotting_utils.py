import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd

def call_plotting_formatting():

    font = {'family' : 'sans-serif',
    'weight' : 'medium',
    'size'   : 15,
    'variant' : 'normal',
    'style' : 'normal',
    'stretch' : 'normal',
    }

    xtick = {'top' : True,
            'bottom' : True,
            'major.size' : 7,
            'minor.size' : 4,
            'major.width' : 0.5,
            'minor.width' : 0.35,
            'direction' : 'in',
            'minor.visible' : True,
            'color' : 'black',
            'labelcolor' : 'black'
            }

    ytick = {'left' : True,
            'right' : True,
            'major.size' : 7,
            'minor.size' : 4,
            'major.width' : 0.5,
            'minor.width' : 0.35,
            'direction' : 'in',
            'minor.visible' : True,
            'color' : 'black',
            'labelcolor' : 'black'
            }

    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['figure.figsize'] = (6.973848069738481, 4.310075139476229)
    mpl.rcParams['figure.subplot.hspace'] = 0.01

    mpl.rc('font', **font)
    mpl.rc('xtick', **xtick)
    mpl.rc('ytick', **ytick)
    mpl.rcParams['legend.fontsize'] = 18
    mpl.rcParams["font.sans-serif"] = ["DejaVu Serif"]
    mpl.rcParams['mathtext.fontset']='dejavuserif'
    mpl.rcParams["text.usetex"] = False

def generate_color_list(num_colors, cmap):

    cmap = plt.get_cmap(cmap)
    colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

    return colors

def plot_stars_data(dfs: list, features: list[str], RANGE=None):

    # Initialize color list for the different dataframes
    colors = [mpl.cm.tab10(i/len(dfs)) for i in range(len(dfs))]
    

    if RANGE is None:
        dfs_all = pd.concat(dfs)
        RANGE = [(dfs_all[f].quantile(0.01), dfs_all[f].quantile(0.99))
                 for f in features]

    for i, df in enumerate(dfs):

        if i==0:
            fig = corner.corner(df[features].values,
                                color=colors[i],
                                labels=["$\log(E)$", "$\log(L)$", "[Fe/H]", "[Mg/Fe]", "E_ERR", "L_ERR", "FeH_ERR", "MgFe_ERR"],
                                bins=20,
                                plot_contours=True,
                                plot_datapoints=False,
                                fill_contours=True,
                                hist_kwargs={"density": True},
                                alpha=0.5,
                                range=RANGE,
                                label_kwargs={'fontsize': 18},
                                )
        else:
            corner.corner(df[features].values,
                            color=colors[i],
                            bins=20,
                            plot_contours=True,
                            plot_datapoints=False,
                            fill_contours=True,
                            hist_kwargs={"density": True},
                            alpha=0.5,
                            range=RANGE,
                            label_kwargs={'fontsize': 18},
                            fig=fig)
            
    return fig


def plot_ax(x,y,
            ax,
            bin_number=500,
            extent=None,
            cmap='cividis'):
            

    counts,x_edges,y_edges = np.histogram2d(x, y, 
                                            bins=(bin_number,bin_number), 
                                            range=extent)
    
    norm = mpl.colors.LogNorm(vmin=1, vmax=200)

    plot = ax.imshow(counts.T,
              cmap=cmap,
              norm=norm,
              origin='lower',
              extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
              interpolation='nearest',
              aspect='equal')


    return plot

def plot_2D_heatmap(x,y,xlabel,ylabel,filename,
                    title=None,
                    weights=None,
                    weighted=False,
                    density=False,
                    cbar_label=r"Number of Stars",
                    bin_number=500,
                    extent=None,
                    cmap='cividis',
                    aspect_ratio=None,
                    norm=None,
                    return_figure=False):
            
    fig,ax = plt.subplots(figsize=(4,4))

    counts,x_edges,y_edges = np.histogram2d(x, y, 
                                            bins=(bin_number,bin_number), 
                                            range=extent,
                                            weights=weights,
                                            density=density)
    
    if weighted:
        nw_counts, _, __ = np.histogram2d(x, y, 
                                          bins=(bin_number,bin_number), 
                                          range=extent,
                                          weights=None)
        nw_counts[nw_counts==0]=-1
        counts = counts/nw_counts
        counts[nw_counts==-1] = np.nan

    if norm is None:
        norm = mpl.colors.LogNorm(vmin=1, vmax=np.percentile(counts, 99.9))

    plot = ax.imshow(counts.T,
                     cmap=cmap,
                     norm=norm,
                     origin='lower',
                     extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
                     interpolation='nearest',
                     aspect='equal')

    cbar = plt.colorbar(plot, orientation='vertical', shrink=0.8)
    cbar.set_label(cbar_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if not aspect_ratio:
        if extent:
            ax.set_aspect((extent[0][1]-extent[0][0])/(extent[1][1]-extent[1][0]))

    if title:
        ax.set_title(title, fontsize=14)

    if return_figure:
        return fig,ax
    else:
        fig.savefig(filename+'.png')
        plt.close('all')

    return 

# Define markers for the different substructures
def generate_markers(n):
    """Generates a list of n unique markers for Matplotlib scatter plots."""
    marker_options = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'X', 'h', 'H', '+', 'x', '|', '_']
    
    # If n is larger than available markers, cycle through them
    markers = list(itertools.islice(itertools.cycle(marker_options), n))
    
    return markers
