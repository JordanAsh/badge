import matplotlib
matplotlib.use('Agg')
import glob
import re
import pandas as pd
from matplotlib import pyplot as plt
from param_utils import param_to_str_fn, param_to_str_t, t_fn
from math import sqrt
import seaborn as sns
import argparse
import numpy as np
import os
from collections import OrderedDict
import pickle
from itertools import accumulate
from math import log2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
import statsmodels.stats.weightstats as wstats
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

foi_pat = '\d+(?:\.\d+)?'
title_size = 33
label_size = 33
tick_size = 26
fig_w = 11.5
fig_h = 7

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def pop_std(x):
    return x.std(ddof=0)

def filter_alg(df, algs):
    df = df.loc[df['Alg'].isin(algs)]
    return df

def parse_file(f, setting, table_dir):
    fh = open(f)

    for content in fh:
        matched_test_acc = test_acc_rgx.findall(content)
        if matched_test_acc:
            #print(matched_test_acc)
            acc = float(matched_test_acc[0][1])
            samples = int(matched_test_acc[0][0])

            if (samples - 100) % setting[3] != 0:
                print(samples)
                continue

            matched_run_time = run_time_rgx.findall(content)
            if matched_run_time:
                rtime = float(matched_run_time[0])
            else:
                rtime = -1

            table_dir.append(setting + [samples, acc, rtime])

    fh.close()

def parse_philly_dirs(dirs, res_table, ovr_cache):
    return parse_dirs(dirs, res_table, '/*/*.txt', "_.*_(.*)_(.*)_(\d+)_([^_]*)_", ovr_cache)

def parse_dirs(dirs, res_table, subdir, res_file_pat, ovr_cache):
    for dir in dirs:
        table_dir = parse_dir(dir, subdir, res_file_pat, ovr_cache)
        res_table += table_dir
    return res_table

def parse_dir(dir, subdir, res_file_pat, ovr_cache):
    if not ovr_cache:
        if os.path.exists(dir+'/cache.p'):
            return pickle.load( open( dir+'/cache.p', 'rb' ) )

    files = sorted(glob.glob(dir+subdir))
    print(len(files))
    table_dir = []

    res_file_rgx = re.compile(res_file_pat, flags=re.M)

    for f in files:
        matched = res_file_rgx.findall(f)
        if matched:
            data, alg, nquery, model = matched[0]
            rep = 0
            train_aug = 0
            nquery = int(nquery)
            setting = [data, model, alg, nquery, train_aug, rep]
            parse_file(f, setting, table_dir)

    pickle.dump( table_dir, open( dir+'/cache.p', 'wb' ) )
    return table_dir


'''
A function that checks that for each setting, how many repeated runs have been done based on the output files.
'''
def plot_count(df):
    df_ct = df.groupby(['Data', 'Model', 'Alg', 'nQuery', 'TrainAug', 'Samples']).agg({'Accuracy': ['count']}).reset_index()
    df_ct['Count'] = df_ct['Accuracy']['count']

    setting_col = ['Data', 'Model', 'nQuery', 'TrainAug']

    for setting, g_setting in df_ct.groupby(setting_col):
        setting_fn = param_to_str_fn(OrderedDict(zip(setting_col, setting)))
        setting_t = param_to_str_t(OrderedDict(zip(setting_col, setting)))

        plt.figure()
        for alg, g_alg in g_setting.groupby(['Alg']):
            ns = list(g_alg['Samples'])
            cts = list(g_alg['Count'])
            cts = [c + np.random.rand() * 0.01 for c in cts]

            plt.plot(ns, cts, label=str(name_dict[alg]), linewidth=1.0, color=color_dict[alg])

        plt.title(setting_t, fontsize=title_size)
        plt.xlabel('Sample size', fontsize=label_size)
        plt.ylabel('Count', fontsize=label_size)
        #plt.legend(frameon=False)
        plt.gca().set_ylim([0,10])

        plt.savefig(all_dir + '/'  + 'count' + '_' + setting_fn + '_' + '.pdf')

'''
Plotting all learning curves
'''
def plot_lc(df, y, tag, highlight):
    sns.set()
    sns.set_style("white")
    #sns.set_context("paper")

    y_mean = y + '_mean'
    y_std = y + '_std'

    avg_folds = df.groupby(['Data', 'Model', 'Alg', 'nQuery', 'TrainAug', 'Samples']).agg(
        {y: ['mean', pop_std, 'count']}).reset_index()
    avg_folds[y_mean] = avg_folds[y]['mean']
    avg_folds[y_std] = avg_folds.apply(lambda row: row[y]['pop_std']/sqrt(row[y]['count']), axis=1)
    avg_folds = avg_folds.drop([y], axis=1)

    print(avg_folds)

    setting_col = ['Data', 'Model', 'nQuery', 'TrainAug']

    for setting, g_setting in avg_folds.groupby(setting_col):
        data, model, nQuery, TrainAug = setting
        setting_t = param_to_str_t(OrderedDict(zip(setting_col, setting)))
        setting_fn = param_to_str_fn(OrderedDict(zip(setting_col, setting)))

        df_rand = g_setting[g_setting['Alg'] == 'rand']

        if nQuery != 10000 and highlight:
            for samp in sorted(list(df_rand['Samples'])):
                acc_vals = g_setting[g_setting['Samples'] == samp][y_mean].values
                #print(max(acc_vals) / min(acc_vals))
                if max(acc_vals) / min(acc_vals) > 1.08:
                    break
                st_samp = samp

            stopFrac = 0.99

            endSamp = max(df_rand['Samples'])
            endAcc = np.mean(df_rand[df_rand['Samples'] == endSamp][y_mean].values)

            for samp in sorted(list(df_rand['Samples'])):
                acc = np.mean(df_rand[df_rand['Samples'] == samp][y_mean].values)
                if acc / endAcc > stopFrac:
                    cut_samp = samp
                    break

            g_setting = g_setting[(g_setting['Samples'] <= cut_samp) & (g_setting['Samples'] >= st_samp)]
        else:
            algs = g_setting['Alg'].unique()
            cut_samp = 1e+8
            st_samp = -1

            for a in algs:
                df = g_setting[g_setting['Alg'] == a]
                mx = max(df['Samples'])
                mn = min(df['Samples'])
                cut_samp = min(cut_samp, mx)
                st_samp = max(st_samp, mn)

        plt.figure(figsize=(fig_w,fig_h))
        for alg, g_alg in g_setting.groupby(['Alg']):
            ns = list(g_alg['Samples'])
            accs = list(g_alg[y_mean])
            stds = list(g_alg[y_std])

            if float(setting[2]) < 10.1:
                accs = smooth(accs, 10)[:-10]
                ns = ns[:-10]
                stds = stds[:-10]

            plt.plot(ns, accs, label=str(name_dict[alg]), linewidth=1.0, color=color_dict[alg], zorder=10-order_dict[alg])
            acc_up = [avg + ci for avg, ci in zip(accs, stds)]
            acc_dn = [avg - ci for avg, ci in zip(accs, stds)]
            plt.fill_between(ns, acc_up, acc_dn, alpha=0.2, color=color_dict[alg], zorder=10-order_dict[alg])

        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.gca().yaxis.get_offset_text().set_fontsize(tick_size)
        if y == 'Time':
            plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.title(setting_t, fontsize=title_size)
        plt.xlabel('#Labels queried', fontsize=label_size)
        plt.subplots_adjust(bottom=0.15)
        plt.ylabel(y, fontsize=label_size)
        plt.gca().set_xlim([st_samp, cut_samp])
        #plt.legend(frameon=False)
        plt.grid(linestyle='--', linewidth=1)
        n_alg = len(g_setting['Alg'].unique())
        save_legend(n_alg)
        plt.savefig(all_dir + '/'  + tag + '_' + y + '_' + setting_fn + '_' + '.pdf')

'''
Plotting all comparison matrices
'''
def plot_all_matrices(df):

    datasets = ['MNIST','SVHN','CIFAR10','155','156','6','184']
    models = ['mlp','rn','vgg']
    nQueries = [100, 1000, 10000]
    algs = ['badge', 'albl', 'coreset', 'conf', 'marg', 'entropy', 'rand']

    plot_matrix(df, datasets, models, nQueries, algs, 'Overall', 'overall')

    for n in nQueries:
        title_tag, tag = t_fn(OrderedDict([('nQuery', n)]))
        plot_matrix(df, datasets, models, [n], algs, title_tag, tag)

    for m in models:
        title_tag, tag = t_fn(OrderedDict([('Model', m)]))
        plot_matrix(df, datasets, [m], nQueries, algs, title_tag, tag)

    '''
    for n in nQueries:
        for m in models:
            title_tag, tag = t_fn(OrderedDict([('nQuery', n), ('Model', m)]))
            plot_matrix(df, datasets, [m], [n], algs, title_tag, tag)
    '''

'''
Plot a comparison matrix in one setting
'''
def plot_matrix(df, datasets, models, nQueries, algs, title_tag, tag):
    df = df.loc[df['Data'].isin(datasets) & df['Model'].isin(models) & df['nQuery'].isin(nQueries)]

    matrix = {}
    for a1 in algs:
        matrix[a1] = {}
        for a2 in algs:
            matrix[a1][a2] = 0

    stopFrac = 0.99

    setting_col = ['Data', 'Model', 'nQuery', 'TrainAug']
    max_poss_ent = 0

    for setting, g in df.groupby(setting_col):
        (data, model, nQuery, train_aug) = setting
        print(data, model, nQuery)
        max_poss_ent += 1
        randRunAv = g[g['Alg'] == 'rand']
        if len(randRunAv['Samples']) == 0: continue
        endSamp = max(randRunAv['Samples'])
        endAcc = np.mean(randRunAv[randRunAv['Samples'] == endSamp]['Accuracy'].values)
        print('endAcc = ', endAcc)
        checkpoints = [100]
        nCheckpoints = 1
        samp = 100
        while True:
            samp = 100 + pow(2.0, nCheckpoints-1) * int(nQuery)
            if samp > endSamp: break
            acc = np.mean(randRunAv[randRunAv['Samples'] == samp]['Accuracy'].values)
            print('acc = ', acc)
            if acc / endAcc > stopFrac: break
            checkpoints.append(samp)
            nCheckpoints += 1

        for checkpoint in checkpoints:
            for alg1 in algs:
                for alg2 in algs:
                    if alg1 == alg2: continue
                    res1 = g[g['Alg'] == alg1]
                    res2 = g[g['Alg'] == alg2]
                    res1[res1['Samples'] == float(checkpoint)]
                    exp1 = res1[res1['Samples'] == float(checkpoint)]['Accuracy'].values
                    exp2 = res2[res2['Samples'] == float(checkpoint)]['Accuracy'].values
                    N1 = len(exp1)
                    N2 = len(exp2)
                    #if N1 < 5: print(data, model, nQuery, alg1, ' has only ' , str(len(exp1)),' runs at checkpoint ', str(checkpoint), '!')
                    if (N1 <= 1) or (N2 <= 1): continue

                    N = min(min(N1, N2), 5)
                    Z = exp1[:N] - exp2[:N]
                    mu = np.mean(Z)
                    var = 1. / (N - 1) * np.sum((Z - mu) ** 2)
                    #t, pval = stats.ttest_ind(exp1[:N], exp2[:N])
                    #z, pval = wstats.ztest(Z / var)
                    t, pval = stats.ttest_1samp(Z, 0.0)
                    if mu < 0 and pval < 0.05:
                        matrix[alg1][alg2] += 1./nCheckpoints
    c = 0
    matPlot = np.zeros((len(matrix), len(matrix)))
    for a1 in algs:
        r = 0
        for a2 in algs:
            matPlot[r][c] = matrix[a1][a2]
            r += 1
        c += 1

    col_avg = matPlot.mean(axis=0)

    matPlot = np.round(matPlot * 100) / 100.
    col_avg = np.round(col_avg * 100) / 100.
    min_e = matPlot.min()
    max_e = matPlot.max()

    plt.rcParams["axes.grid"] = False
    fig, ax = plt.subplots()

    ax.tick_params(axis=u'both', which=u'both',length=0)
    im = ax.matshow(matPlot, cmap='viridis', vmin=min_e, vmax=max_e)

    #ax.set_xticks(np.arange(len(algs)))
    #ax.set_yticks(np.arange(len(algs)))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels([0] + [name_dict[alg] for alg in algs], fontsize=8)
    ax.set_yticklabels([0] + [name_dict[alg] for alg in algs], rotation=0, fontsize=8)

    for i in range(len(algs)):
        for j in range(len(algs)):
            text = ax.text(j, i, matPlot[i, j], ha="center", va="center", color="w", fontsize=8)

    ax.set_title(title_tag + '({})'.format(max_poss_ent))
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size=1, pad=-0.25)

    for j in range(len(algs)):
        text = ax2.text(j, 0, col_avg[j], ha="center", va="center", color="w", fontsize=8)

    im = ax2.matshow(np.array([col_avg]), cmap='viridis', vmin=min_e, vmax=max_e)
    ax2.axis('off')

    fig.subplots_adjust(right=1)
    cbar_ax = fig.add_axes([0.8, 0.18, 0.03, 0.7])
    cbar = fig.colorbar(im,cax=cbar_ax)
    cbar.outline.set_visible(False)
    plt.savefig(all_dir + '/' + 'comp_matrix' + '_' + tag + '.pdf')
    plt.rcParams["axes.grid"] = True

def cdf(alg, errs, iw):
    idx = np.argsort(errs)
    num_errs = len(errs)
    sorted_errs = [errs[idx[i]] for i in range(num_errs)]
    sorted_iw = [iw[idx[i]] for i in range(num_errs)]

    plt.step(sorted_errs, np.cumsum(sorted_iw), label=name_dict[alg], color=color_dict[alg], linewidth=2.0, zorder=10-order_dict[alg])

def avg_folds(df):
    avg = df.groupby(['Data', 'Model', 'Alg', 'nQuery', 'TrainAug', 'Samples']).agg('mean').reset_index()
    return avg

def plot_sep_cdfs(df):
    datasets = ['MNIST','SVHN', 'CIFAR10', '6', '155', '156', '184']
    models = ['mlp','rn', 'vgg']
    nQueries = [100, 1000, 10000]
    algs = ['badge', 'albl', 'coreset', 'conf', 'marg', 'entropy', 'rand']

    plot_cdfs(df, datasets, models, nQueries, algs, 'Overall', 'overall')

    for n in nQueries:
        title_tag, tag = t_fn(OrderedDict([('nQuery', n)]))
        plot_cdfs(df, datasets, models, [n], algs, title_tag, tag)

    for m in models:
        title_tag, tag = t_fn(OrderedDict([('Model', m)]))
        plot_cdfs(df, datasets, [m], nQueries, algs, title_tag, tag)

    for n in nQueries:
        for m in models:
            title_tag, tag = t_fn(OrderedDict([('nQuery', n), ('Model', m)]))
            plot_cdfs(df, datasets, [m], [n], algs, title_tag, tag)

def save_legend(nc):
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    figlegend = plt.figure(figsize=(nc*3,2))
    leg = figlegend.legend(handles, labels, 'center', fontsize=26, ncol=nc)
    for line in leg.get_lines():
        line.set_linewidth(4)
    figlegend.savefig(all_dir+'/legend.pdf')
    plt.close()


def plot_cdfs(df, datasets, models, nQueries, algs, title_tag, tag):
    sns.set()
    sns.set_style("white")
    #sns.set_context("paper")

    n_errs = {a:[] for a in algs}
    iws = []
    algs_set = set(algs)
    stopFrac = 0.99

    for nQuery in nQueries:
        for data in datasets:
            for model in models:
                # figure out when to stop sampling checkpoints
                randRunAv = df[(df['Data'] == data) &
                            (df['Model'] == model) &
                            (df['Alg'] == 'rand') &
                            (df['nQuery'] == nQuery)
                            ]

                if len(randRunAv['Samples']) == 0: continue
                endSamp = max(randRunAv['Samples'])
                endAcc = np.mean(randRunAv[randRunAv['Samples'] == endSamp]['Accuracy'].values)
                checkpoints = [100]
                nCheckpoints = 1
                samp = 100
                while True:
                    samp = 100 + int(pow(2.0, nCheckpoints-1)) * int(nQuery)
                    if samp > endSamp: break
                    acc = np.mean(randRunAv[randRunAv['Samples'] == samp]['Accuracy'].values)
                    if acc / endAcc > stopFrac: break
                    checkpoints.append(samp)
                    nCheckpoints += 1

                for checkpoint in checkpoints:
                    accs = {}
                    skip = False

                    for alg in algs:
                        res = df[(df['Data'] == data) &
                            (df['Model'] == model) &
                            (df['Alg'] == alg) &
                            (df['nQuery'] == nQuery) &
                            (df['Samples'] == checkpoint)]

                        acc = res['Accuracy'].values
                        if len(acc) != 0:
                            accs[alg] = acc[0]
                        else:
                            skip = True
                            break
                    if skip:
                        continue

                    for alg in algs:
                        n_errs[alg].append( (1 - accs[alg]) / (1 - accs['rand']) )

                iws += [1.0 / nCheckpoints for _ in range(nCheckpoints)]

    plt.figure(figsize=(fig_w,fig_h))
    sum_iw = sum(iws)
    iws = [iw / sum_iw for iw in iws]
    for alg in algs:
        cdf(alg, n_errs[alg], iws)

    plt.gca().set_xlim([0.3,1.2])
    plt.title(title_tag, fontsize=title_size)
    plt.xlabel('Normalized error', fontsize=label_size)
    plt.ylabel('Cumulative frequency', fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.subplots_adjust(bottom=0.15)
    plt.grid(linestyle='--', linewidth=1)
    plt.savefig(all_dir + '/' + 'cdf' + '_' + tag + '.pdf')

'''
Result analysis script for plotting:
- learning curves,
- comparison matrices,
- normalized error CDFs

Example command:
python agg_results.py ~/deep_active/philly_configs pull_list.txt output_all 1 0,

Here:
- ~/nycml/deep_active/philly_configs/ is the folder that stores all the results
- pull_list.txt is the names of subdirectories under deep_active/philly_configs/ that stores all
- output_all is the name the of the subdirectory that stores the output figures
- the second-to-last argument '1' means that we overwrite (therefore, do not use) the cache file for the output results; after first run of the command, cache files will be created in each subdirectories, therefore, in subsequent runs, we can set the argument to '0' to directly load parsed and cached data.
- the last argument '0' means that we do not turn on the interactive mode

See also explanations of command line arguments below.
'''

if __name__  == '__main__':
    test_acc_pat = "("+foi_pat+")"+"\s+testing accuracy\s+"+"("+foi_pat+")"
    run_time_pat = "running time\s+"+"("+foi_pat+")"
    test_acc_rgx = re.compile(test_acc_pat, flags=re.M)
    run_time_rgx = re.compile(run_time_pat, flags=re.M)

    algs = ['badge', 'albl', 'coreset', 'conf', 'marg', 'entropy', 'rand', 'baseline']
    alg_name = ['BADGE', 'ALBL', 'Coreset', 'Conf', 'Marg', 'Entropy', 'Rand', 'k-DPP']
    alg_order = range(len(algs))

    name_dict = {a:n for (a, n) in zip(algs, alg_name)}
    order_dict = {a:o for (a, o) in zip(algs, alg_order)}

    palette = sns.color_palette('colorblind', 10)
    p = palette.as_hex()
    colors = [p[0], p[9], p[2], p[1], p[3], p[5], p[4], p[7]]
    color_dict = dict(zip(algs, colors))

    parser = argparse.ArgumentParser()
    parser.add_argument('dir_prefix',help='directory prefix', type=str)
    parser.add_argument('pull_list', help='result directory list (in philly format)', type=str)
    parser.add_argument('fig_dir', help='figure directory', type=str)
    parser.add_argument('ovr_cache', help='whether to overwrite cache file', type=int)
    parser.add_argument('interactive', help='interactive mode to explore the generated file', type=int)
    args = parser.parse_args()
    dir_prefix = args.dir_prefix
    fig_dir = args.fig_dir
    ovr_cache = args.ovr_cache

    pull_list = []
    f = open(dir_prefix + '/' + args.pull_list)
    for i in f:
        print(dir_prefix + '/' + i.strip())
        pull_list.append(dir_prefix + '/' + i.strip())
    f.close()

    all_dir = dir_prefix + '/' + fig_dir
    if not os.path.exists(all_dir):
        os.mkdir(all_dir)

    res_table = []
    parse_philly_dirs(pull_list, res_table, ovr_cache)


    df = pd.DataFrame(res_table, columns=[
                      'Data', 'Model', 'Alg', 'nQuery', 'TrainAug', 'Rep', 'Samples', 'Accuracy', 'Time'])
    print(df)

    df['Samples'] = df['Samples'].astype(int)

    if args.interactive:
        import pdb; pdb.set_trace()

    '''
    Checking the job completion status
    '''
    plot_count(df)

    '''
    For running time comparison between BADGE (w/ k-means) and DPP
    '''
    df1 = df[ (df['Alg'] == 'baseline') | (df['Alg'] == 'badge') ]
    plot_lc(df1, 'Time', 'comp', False)
    plot_lc(df1, 'Accuracy', 'comp', False)

    '''
    For plotting the learning curves of all algorithms
    '''
    df2 = df[df['Alg'] != 'baseline']
    plot_lc(df2, 'Accuracy', 'highlight', True)
    plot_lc(df2, 'Accuracy', 'full', False)

    '''
    Comparison matrices
    '''
    df3 = df[df['Alg'] != 'baseline']
    plot_all_matrices(df3)
    plot_sep_cdfs(df3)
