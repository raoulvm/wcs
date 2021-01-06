from _typeshed import NoneType
from typing import Tuple, List, Union # typed Lists

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.backends.backend_pdf
# %matplotlib inline

def _compute_nb_steps(power: float, dict_diff_steps: dict, vals: list,
                      min_val: float, max_val: float, bins: int) -> dict:
    """
    Helper function to compute the step size for different values (used in *compute_step_size*)
    """
    for val in vals:
        dict_diff_steps[val * power] = np.ceil((max_val - min_val) / (val * power)) - bins
    return dict_diff_steps


def _get_number_of_decimals(num: float, eps: float = 1e-5):
    decimals = -1

    while True:
        decimals += 1
        tmp_num = num * 10 ** decimals
        if abs(tmp_num - np.round(tmp_num)) < eps and tmp_num > 0.5:
            break
    return decimals


def _compute_step_size(min_val: float, max_val: float, bins: int) -> Tuple[float, int]:
    """
    Given a continuous variable, function computes the most appropriate step size for a histogram
    given some 'nice' potential ranges (see below). The idea behind this is the following: Given a variable that takes
    values between 0 and 10 and let's assume we want ~12 bins. One way to go would be to just set the step size
    to 10/12 ~ 0.83, which would end up in pretty strange ranges when presenting results. Our way, however, is to check
    the number of bins which would result in 'nice' ranges. Notice e.g. that a step size of 1 would result in 10
    bins which is pretty close to 12. We would therefore select a step size of 1 and end up with 10 bins.
    :param min_val: (float) Minimal value taken by the variable
    :param max_val: (float) Maximal value taken by the variable
    :param bins: (int) Preferred number of bins in histogram
    :return: float with ideal step size
    """

    # nice values are considered to be *nice_vals* x 10^y
    nice_vals = [1, 2, 3, 4, 5]
    power = 1

    # initialize the dictionary containing the
    dict_steps = {}

    # st
    dict_steps = _compute_nb_steps(power, dict_steps, nice_vals, min_val, max_val, bins)

    while True:
        # if the power is too big, we reduce it
        if dict_steps[nice_vals[0] * power] < 0 and not (nice_vals[0] * power / 10) in dict_steps.keys():
            power /= 10
            dict_steps = _compute_nb_steps(power, dict_steps, nice_vals, min_val, max_val, bins)
        # if the power is too big, we increase it
        elif dict_steps[nice_vals[-1] * power] > 0 and not (nice_vals[-1] * power * 10) in dict_steps.keys():
            power *= 10
            dict_steps = _compute_nb_steps(power, dict_steps, nice_vals, min_val, max_val, bins)
        else:
            break

    # find the step size with which the number of bins is closest to *bins*
    step_size = sorted(dict_steps.items(), key=lambda item: abs(item[1]))[0][0]

    # compute the number of decimals of the step size, e.g. if step_size = 0.04, then decimals = 2
    decimals = _get_number_of_decimals(step_size)

    return step_size, decimals


def _preprocess_continuous_variable(df: pd.DataFrame, var_col: str, bins: int,
                                    min_val: float = None,
                                    max_val: float = None) -> pd.DataFrame:
    """
    Pre-processing the histogram for continuous variables by splitting the variable in buckets.
    :param df: (pd.DataFrame) Data frame containing at least the continuous variable
    :param var_col: (str) Name of the continuous variable
    :param bins: (int) Preferred number of bins in histogram
    :param min_val: (float, optional) Minimal value to be taken by the variable (if other than the minimum observed in
                    the data.
    :param max_val: (float, optional) Maximal value to be taken by the variable (if other than the maximum observed in
                    the data.
    :return: pd.DataFrame with *var_col* transformed to range
    """

    # set *min_val* and *max_val* to minimal and maximal values observed in data
    if min_val is None:
        min_val = df[var_col].min()
    if max_val is None:
        max_val = df[var_col].max()

    # compute the most appropriate step size for the histogram
    step_size, decimals = _compute_step_size(min_val, max_val, bins)

    min_val = min_val - (min_val % step_size)

    # cut values into buckets
    df[var_col] = pd.cut(df[var_col],
                         list(np.arange(min_val, max_val, step_size)) + [max_val],
                         include_lowest=True)

    # convert buckets into strings
    if decimals == 0:
        df[var_col] = df[var_col].map(lambda x: f"{int(np.round(x.left))} - {int(np.round(x.right))}")
    else:
        df[var_col] = df[var_col].map(lambda x: f"{np.round(x.left, decimals)} - {np.round(x.right, decimals)}")
    return df


def _prepare_histogram_line_chart(df: pd.DataFrame,
                                 var_col: str,
                                 target_col: str,
                                 var_col_type: str,
                                 bins: int = 12,
                                 min_val: float = None,
                                 max_val: float = None,
                                 pred_col: str = None,
                                 miss_show_percent: float = 1,
                                 residuals:bool=False,
                                 mean_logloss:bool=False,
                                 total_logloss:bool=False,
                                 ) -> pd.DataFrame:
    """
    Prepare the histogram line chart by computing the number of observations and the average target share per bin.
    :param df: (pd.DataFrame) Data frame containing at least the columns *var_col* and *target_col*
    :param var_col: (str) Name of variable to be plotted
    :param target_col: (str) Name of the target variable (currently only binary variables are supported)
    :param var_col_type: (str) Type of the variable, i.e. 'continuous', 'ordinal' or 'categorical'
    :param bins: (int) Preferred number of bins in histogram
    :param min_val: (float, optional) Minimal value to be taken by the variable (if other than the minimum observed in
                    the data.
    :param max_val: (float, optional) Maximal value to be taken by the variable (if other than the maximum observed in
                    the data.
    :param miss_show_percent: (float, default=1) Add a missing bin if more than *miss_show_percent* percent of the
                              values are missing
    :return: pd.DataFrame with number of observation and average target share per bin
    """
    if var_col_type not in ["continuous", "ordinal", "categorical"]:
        raise ValueError("*var_col_type* can only take values 'continuous', 'ordinal' or 'categorical'")

    if var_col not in df.columns:
        raise ValueError("*var_col* needs to be column of *df*")

    if target_col not in df.columns:
        raise ValueError("*target_col* needs to be column of *df*")

    if len(df[target_col].unique()) > 2:
        raise ValueError("Function currently only supports binary classification")

    df_miss = df[df[var_col].isnull()].copy()
    df = df[df[var_col].notnull()].copy()

    # clip min and max values if necessary
    if var_col_type in ["continuous", "ordinal"]:
        if min_val is not None:
            df[var_col] = df[var_col].clip(lower=min_val)

        if max_val is not None:
            df[var_col] = df[var_col].clip(upper=max_val)

    # pre-process the continuous variable by putting the values into bins
    if var_col_type == "continuous":
        df = _preprocess_continuous_variable(df, var_col, bins, min_val, max_val)
    
    # if there is no pred columns we manually build one
    if pred_col is None:
      drop_pred = True
      pred_col = "xyz_pred_xyz"
      df[pred_col] = 0
    else: 
      drop_pred = False

    # put the values into bins and calculate the mean target share per bin

    residual_col = 'RESIDUUM'
    logloss_col = 'LOGLOSS'

    if pred_col = "xyz_pred_xyz":
        # dummy only
        df2 = df.assign(**{residual_col: 0,
                    logloss_col: 0
                    })
    else:
        df2 = df.assign(**{residual_col: df[target_col]-df[pred_col],
                        logloss_col: -(df[target_col]*np.log(df[pred_col]) + (1 - df[target_col]) * np.log(1 - df[pred_col]))
                        })
    

    df_agg = pd.pivot_table(df2, index=var_col,
                            aggfunc={var_col: "count", 
                                     target_col: [np.mean], #, np.std],
                                     pred_col: [np.mean], #, np.std]
                                     residual_col: [np.mean],
                                     logloss_col: [np.mean, np.sum]

                                     })

    # get rid of the multilevel index    
    df_agg.columns = df_agg.columns.ravel()

    df_agg.rename(columns={(var_col,'count'): "COUNT", 
                           (target_col,'mean'): "SHARE_TARGET_PERC",
                           #(target_col, 'std'): "SHARE_TARGET_STD",
                           (pred_col, 'mean'): "MEAN_PREDICTION",
                           #(pred_col, 'std'): "PREDICTION_STD",
                           (residual_col, 'mean'): "MEAN_RESIDUUM",
                           (logloss_col, 'mean'): "MEAN_LOGLOSS",
                           (logloss_col, 'sum'): "TOTAL_LOGLOSS",
                           }, inplace=True)

    df_agg.reset_index(inplace=True)

    #display(df_agg.columns)



    # make sure we are talking about percentages
    df_agg["SHARE_TARGET_PERC"].fillna(0, inplace=True)

    df_agg["MEAN_PREDICTION"].fillna(0, inplace=True)

    df_agg["MEAN_RESIDUUM"].fillna(0, inplace=True)

    # sort the values
    if var_col_type == "categorical":
        df_agg.sort_values("SHARE_TARGET_PERC", inplace=True)
    else:
        df_agg.sort_values(var_col, inplace=True)

    df_agg.reset_index(drop=True, inplace=True)

    # in case of a continuous variable, we want to write <= 1 instead of e.g. [0,1] for the first bin and
    # > 10 instead of e.g. (10, 23] for the last bin
    if var_col_type == "continuous":
        df_agg[var_col] = df_agg[var_col].astype("string")
        val = df_agg.iloc[0][var_col]
        df_agg.loc[0, var_col] = "<= " + val.split(" - ")[-1]

        val = df_agg.iloc[-1][var_col]
        df_agg.iloc[-1, df_agg.columns.get_loc(var_col)] = x = "> " + val.split(" - ")[0]

    if drop_pred:
      df_agg = df_agg.drop(columns=['MEAN_PREDICTION',
                                    'MEAN_RESIDUUM',
                                    'MEAN_LOGLOSS', 
                                    'TOTAL_LOGLOSS',
                                    ]).copy()
    else:
      df_agg = df_agg.copy()

    if len(df_miss) > (miss_show_percent / 100 * (len(df_miss) + len(df))):
        miss_line = ["Missing", len(df_miss), np.mean(df_miss["TARGET"]) * 100]
        df_agg.loc[len(df_agg)] = miss_line

    return df_agg


def plot_histogram_with_target(df: pd.DataFrame,
                               var_col: str,
                               target_col: str,
                               var_col_type: str,
                               pred_col: str = None,
                               bins: int = 12,
                               min_val: float = None,
                               max_val: float = None,
                               return_df: bool = False,
                               residuals:bool=False,
                               mean_logloss:bool=False,
                               total_logloss:bool=False,
                               fix_ratio_scale:float=1,
                               pdf_object:matplotlib.backends.backend_pdf.PdfPages=None,
                               ):
    """
    Create a histogram plot of the variable *var_col* combined with the average value of *target_col* per bucket
    in the histogram.
    :param df: (pd.DataFrame) Data frame containing at least the columns *var_col* and *target_col*
    :param var_col: (str) Name of variable to be plotted
    :param target_col: (str) Name of the target variable (currently only binary variables are supported)
    :param var_col_type: (str) Type of the variable, i.e. 'continuous', 'ordinal' or 'categorical'
    :param full_path: (str) Path where the picture should be stored
    :param bins: (int) Preferred number of bins in histogram
    :param min_val: (float, optional) Minimal value to be taken by the variable (if other than the minimum observed in
                    the data)
    :param max_val: (float, optional) Maximal value to be taken by the variable (if other than the maximum observed in
                    the data)
    :param return_df: (bool, default=False) Whether or not the data frame with the data underlying the plot is returned


    :return: None or pd.DataFrame with the data underlying the plot (only if *return_df* is set to True)
    """
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    if sum(df[var_col].isnull()) > 0:
      raise ValueError(f"Variable {var_col} contains NULL values")

    # compute the data that is needed for the plot
    df_agg = _prepare_histogram_line_chart(df, var_col, target_col, var_col_type,
                                          bins, min_val, max_val, pred_col=pred_col)

    # create the folder to store the plot (if it does not yet exist)
    #if not os.path.exists(full_path):
    #    os.makedirs(full_path)

    # set up the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    barlabel = f'{var_col} count'
    bars  = ax1.bar(list(df_agg.index), list(df_agg["COUNT"]), color='lightgrey',  label=barlabel)
    
    redlinelabel = f'target positive share'
    redline  = ax2.plot(list(df_agg["SHARE_TARGET_PERC"]), color='red', linewidth=3, label=redlinelabel)[0]

    spinepos = 1.0

    if pred_col is not None:
        blacklinelabel =   f'predicted share'
        blackline  = ax2.plot(list(df_agg["MEAN_PREDICTION"]), color='black', linewidth=3, label=blacklinelabel)[0]
        if residuals:
            ax3 = ax1.twinx()
            spinepos += 0.1
            ax3.spines["right"].set_position(("axes", spinepos))
            make_patch_spines_invisible(ax3)
            ax3.spines["right"].set_visible(True)
            ax3.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
            ax3.set_ylabel('Mean Residual')
            bluelinelabel = f'mean error'
            blueline  = ax3.plot(list(df_agg["MEAN_RESIDUUM"]), color='blue', linewidth=3, label=bluelinelabel)[0]
        if mean_logloss:
            ax4 = ax1.twinx()
            spinepos += 0.1
            ax4.spines["right"].set_position(("axes", spinepos))
            make_patch_spines_invisible(ax4)
            ax4.spines["right"].set_visible(True)
            #ax4.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
            ax4.set_ylabel('Mean LogLoss')
            bluebarlabel = f'mean logloss'
            bluebar  = ax4.bar(list(df_agg.index), list(df_agg["MEAN_LOGLOSS"]), color='blue', width=.1, align='edge', alpha=.5, label=bluebarlabel)            
        if total_logloss:
            ax5 = ax1.twinx()
            spinepos += 0.1
            ax5.spines["right"].set_position(("axes", spinepos))
            make_patch_spines_invisible(ax5)
            ax5.spines["right"].set_visible(True)
            #ax5.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
            ax5.set_ylabel('Total LogLoss')
            bluebarlabel = f'total logloss'
            greybar  = ax5.bar(list(df_agg.index), list(df_agg["TOTAL_LOGLOSS"]), color='grey', width=-0.1, align='edge', alpha=.5, label=bluebarlabel)     

    ax1.grid(b=False)

    ax1.set_title(f'"{var_col}" vs. Target "{target_col}"' , fontsize=20)
    ax1.set_ylabel("Observations", fontsize=14)
    ax2.set_ylabel("Target rate", fontsize=14)

    ax2.set_ylim(bottom=0)
    if fix_ratio_scale is not None:
        ax2.set_ylim(top=fix_ratio_scale)

    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    ax1.set_xticks(np.arange(len(df_agg[var_col])))
    labels = list(df_agg[var_col])
    ax1.axes.set_xticklabels(labels, rotation=45)
    #TODO das ist strubbelig und muss aufgeraeumt werden!

    if pred_col is None:
        #plt.legend((bars, redline,), 
        #           (barlabel, redlinelabel,)   )
        lines = [bars, redline,]
    else:
        if residuals and pred_col is not None:
            # plt.legend((bars, redline, blackline, blueline), 
            #            (barlabel, redlinelabel, blacklinelabel, bluelinelabel)   )
            lines = [bars, redline, blackline, blueline]
        else:
            # no residuals
            # plt.legend((bars, redline, blackline, ), 
            #            (barlabel, redlinelabel, blacklinelabel, )   )
            lines = [bars, redline, blackline]
        if mean_logloss:
            lines.append(bluebar)
        if total_logloss:
            lines.append(greybar)
    ax1.legend(lines, [l.get_label() for l in lines])
    
    # save the plot
    fname_out = f"{var_col.replace(' ', '_').lower()}.png"
    plt.tight_layout()
    #plt.savefig(os.path.join(full_path, fname_out))
    if pdf_object is not None:
        pdf_object.savefig(fig)
        fig.clf()
        plt.close()
    else:
        plt.show()
    if return_df:
        return df_agg


def build_histograms_with_target(df: pd.DataFrame, target_col: str, cont_cols: List[str], ord_cols: List[str], cat_cols: List[str],
                                 pred_col: str = None, perc_winsor: float = 1, error:str=None,  fix_ratio_scale:Union[float,NoneType]=1, as_pdf_name:str=None):
    """
    For each variable, create a histogram in which for each bin of the histogram the average target value is plotted
    as well.
    :param df: (pd.DataFrame) Data frame containing at least all columns specified in *cont_col*, *ord_cols* and
               *cat_cols* and also the *target_col*
    :param target_col: (str) Name of the target variable (currently only binary variables are supported)
    :param cont_cols: (list) List of all columns with continuous variables
    :param ord_cols: (list) List of all columns with ordinal or boolean variables
    :param cat_cols: (list) List of all columns with categorical variables
    :param pred_col: (str) Name of the column with the currently predicted percentages
    :param perc_winsor: (float) For continuous variables the variable is winsorized with *perc_winsor* and
                        100 - *perc_winsor* before building the buckets for the histogram
    :param error: (str) None, 'residuals', 'meanlogloss', 'totallogloss','all'
    :param fix_ratio_scale: (float) fixates the top value of the ratio scale to the given number. None to AutoScale for each chart indivudually
    :param as_pdf_name: (str) Saves the created charts in a pdf to the file system.

    :return: None, but prints/saves histograms
    """
    calc_residuals:bool=False

    if error in ['residuals', 'all']:
        calc_residuals=True
    
    mean_logloss:bool=False
    if error in ['meanlogloss', 'all']:
        mean_logloss=True

    total_logloss:bool=False
    if error in ['totallogloss', 'all']:
        total_logloss=True

    print("Plotting continuous variables")
    print("#############################")


    pdf = None
    if as_pdf_name is not None:
        # create a pdf file with all the graphs for offline use
        pdf = matplotlib.backends.backend_pdf.PdfPages(as_pdf_name)

    for col in cont_cols:
        print(col)
        df_avail = df[df[col].notnull()].copy()
        min_val = np.percentile(df_avail[col], perc_winsor)
        max_val = np.percentile(df_avail[col], 100 - perc_winsor)

        plot_histogram_with_target(df, col, target_col,
                                   "continuous", pred_col=pred_col,
                                   min_val=min_val, max_val=max_val, 
                                   residuals = calc_residuals, 
                                   total_logloss=total_logloss,
                                   mean_logloss=mean_logloss,
                                   fix_ratio_scale=fix_ratio_scale, 
                                   pdf_object=pdf)

    print("Plotting ordinal variables")
    print("##########################")
    for col in ord_cols:
        print(col)
        plot_histogram_with_target(df, col, target_col, "ordinal", 
                                   pred_col=pred_col, 
                                   residuals = calc_residuals, 
                                   total_logloss=total_logloss,
                                   mean_logloss=mean_logloss,
                                   fix_ratio_scale=fix_ratio_scale, 
                                   pdf_object=pdf)

    print("Plotting categorical variables")
    print("##############################")
    for col in cat_cols:
        print(col)
        plot_histogram_with_target(df, col, target_col, "categorical", 
                                   pred_col=pred_col, 
                                   residuals = calc_residuals, 
                                   total_logloss=total_logloss,
                                   mean_logloss=mean_logloss,                                   
                                   fix_ratio_scale=fix_ratio_scale, 
                                   pdf_object=pdf)
    if pdf is not None:
        pdf.close()