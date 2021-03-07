# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 
import glob
import os
import pandas as pd
import matplotlib.gridspec as gridspec
import glob
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json


def return_curve_autoname(save_dir="./output/",
                          env_name="sparse-point-robot",
                        algo_name="FOCAL",
                        prefix=None,
                        **kwargs
                        ):
        exp_dict = {}
        root_dir = os.path.join(save_dir, env_name)
        for fn in os.listdir(root_dir):
            variant_path = os.path.join(root_dir, fn, "variant.json")
            progress_path = os.path.join(root_dir, fn, "progress.csv")
            print(prefix, fn)
            if prefix is not None and prefix not in fn:
                continue
            if os.path.exists(variant_path) and os.path.exists(progress_path):
                variant = json.load(open(variant_path, 'r'))
                try:
                    progress = pd.read_csv(progress_path)
                except:
                    continue
                if not progress.empty:
                    algo_params = variant["algo_params"]
                    T = algo_params["T"]
                    agent_type = algo_params["agent_type"]
                    m = algo_params["m"]
                    use_brac = algo_params["use_brac"]
                    train_epoch = algo_params["train_epoch"]
                    use_transformer_batch = algo_params["use_transformer_batch"]
                    use_transformer_sequence = algo_params["use_transformer_sequence"]
                    use_multihead_attention = algo_params["use_multihead_attention"]
                    use_channel_attention = algo_params["use_channel_attention"]
                    key = f"{algo_name}_{agent_type}_T{T}_m{m}_brac{use_brac}_epoch{train_epoch}_batch{use_transformer_batch}_sequence{use_transformer_sequence}_channel{use_channel_attention}_multihead{use_multihead_attention}"
                    if key in exp_dict.keys():
                        exp_dict[key].append(progress_path)
                    else:
                        exp_dict[key] = [progress_path]

        print('kwargs', kwargs)
        print('exp_dict', exp_dict)
        return_curve(save_dir=save_dir,
                     env_name=env_name,
                     **{**kwargs, **exp_dict})


def return_curve(
               data_dir_lst,
               env_name_lst,
               save_dir="./output/figures",
               env_name="sparse-point-robot",
               task_name=None,
               train_tasks=80,
               test_tasks=20,
               n_columns=9,
               alpha=0.1,
               xlim=None,
               ylim=None,
               xlabel='Sample Steps',
               ylabel=None,
               title=None,
               aggre_step=1,
               grid=1,
               sci_label=1,
               fig_keys=None,
               fontsize=15,
               save=False,
               **kwargs):

    if fig_keys is None:
        df = pd.read_csv(list(kwargs.values())[0][0])
        fig_keys = df.columns[:-n_columns]
    n_color = len(env_name_lst)


    colormap = cm.gist_ncar #nipy_spectral, Set1,Paired  
    colorst = [colormap(i) for i in np.linspace(0, 0.9, n_color)] 

    result_df = pd.DataFrame()
    for fig_key in fig_keys:
        for c, key, dir_lst in zip(colorst, env_name_lst, data_dir_lst):
            y_lst = []
            if len(dir_lst) > 1:
                min_dim = 99999
                for dir in dir_lst:
                    print(dir)
                    df = pd.read_csv(dir)
                    x = df['Number of env steps total'].to_numpy()
                    if x.shape[0] < min_dim:
                        min_dim = x.shape[0]
                    try:
                        y = df[fig_key].to_numpy().reshape((1, -1))
                        y_lst.append(y)
                    except:
                        pass
                for i, _ in enumerate(y_lst):
                    y_lst[i] = y_lst[i][:, :min_dim]
                x = x[:min_dim]
                if len(y_lst) == 0: # no data, skip plotting
                    continue


                y = np.concatenate(y_lst, axis=0)

                if aggre_step > 1:
                    dim1 = int(int((y.shape[1]-1)/aggre_step) * aggre_step) # effective dimension of axis 1 for aggregate plotting
                    dim0 = y.shape[0]
                    y_data = y[:, 1:1+dim1] # do not average the initial value of x and y
                    y_data = y_data.reshape((dim0, -1, int(aggre_step)))
                    y_data = y_data.transpose((0, 2, 1)).reshape((dim0 * aggre_step, -1))
                    y = np.concatenate([y[: 0], y])
                    x_data = x[1:1+dim1]
                    x_data = x_data.reshape((-1, int(aggre_step)))
                    print(x_data, x_data.shape)
                    x_data_mean = np.mean(x_data, axis=1)
                    y_data_mean = np.mean(y_data, axis=0)
                    y_data_std = np.std(y_data, axis=0)

                    print(x[0:1].shape, x_data_mean.shape)
                    x = np.concatenate([x[0:1], x_data_mean], axis=0)
                    y_mean = np.concatenate([np.mean(y[:, 0:1], axis=0), y_data_mean], axis=0)
                    y_std = np.concatenate([np.std(y[:, 0:1], axis=0), y_data_std], axis=0)
                    print(y.shape, x.shape)


                else:
                    y_mean = np.mean(y, axis=0)
                    y_std = np.std(y, axis=0)
                l, = plt.plot(x, y_mean, label=key, linewidth=0.5, color=c)


                plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=alpha, color=c)

                result_df = result_df.append({'mean': np.mean(y_mean[-5:]), 
                                  'std': np.mean(y_std[-5:]),
                                  'path': dir_lst[0],
                                  'key': key}, ignore_index=True)

            else:
                df = pd.read_csv(dir_lst[0])
                x = df['Number of env steps total'].to_numpy()
                y = df[fig_key].to_numpy()
                plt.plot(x, y, label=key, linewidth=0.5, color=c)
                result_df = result_df.append({'mean': np.mean(y_mean[-5:]), 
                                  'std': np.mean(y_std[-5:]),
                                  'path': dir_lst[0],
                                  'key': key}, ignore_index=True)


        plt.xlabel(xlabel)
        if ylabel is None:
            if "AverageReturn_all_test_tasks" in fig_key:
                plt.ylabel("Average Return", fontsize=fontsize)
                plt.xlabel("Sample Steps", fontsize=fontsize)
            else:
                plt.ylabel(fig_key)
        else:
            plt.ylabel(ylabel)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        plt.legend(prop={'size': fontsize}, framealpha=0.3, fontsize=fontsize)

        if grid:
            plt.grid(linestyle='--', linewidth=0.8)
        if sci_label:
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

        if title is None:
            plt.title(env_name.split('.')[0] + f', train_tasks={train_tasks}, test_tasks={test_tasks}', fontsize=fontsize)
        else:
            plt.title(title, fontsize=fontsize)
        
        os.makedirs(save_dir, exist_ok=True)

        if save:
            plt.savefig(os.path.join(save_dir, f'{env_name}_{fig_key}.png'), dpi=200)
            plt.close()
            result_df = result_df.sort_values(by=['mean'])
            result_df.to_csv(os.path.join(save_dir, f'{env_name}_{fig_key}.csv'))

def plot_baseline(data_dir_lst, ax, label, xlim=[0, 1.e6], alpha=0.1, xlog=False):
    mean_lst = [np.load(data_dir)['mean'] for data_dir in data_dir_lst]
    std_lst = [np.load(data_dir)['std'] for data_dir in data_dir_lst]
    mean_npy = np.array(mean_lst)
    std_npy = np.array(std_lst)
    import os
    iter = []
    if len(mean_lst[0]) == 19:
        ratio = 200
    else:
        ratio = 100
    for i in range(1, len(mean_lst[0]) + 1):
        iter.append(i * 256 * ratio)

    mean = np.mean(mean_npy, axis=0)
    std = np.sqrt(np.mean(std_npy ** 2, axis=0))
    l,  = ax.plot(iter, mean, label=label)
    r1 = list(map(lambda x: x[0] - x[1], zip(mean, std)))
    r2 = list(map(lambda x: x[0] + x[1], zip(mean, std)))
    ax.fill_between(iter, r1, r2, color=l.get_color(), alpha=alpha)

    ax.set_xlabel('Sample Steps')
    ax.set_ylabel('Average Return')
    ax.set_xlim(xlim)
    if xlog:
        ax.set_xscale('log')
    handles, labels = ax.get_legend_handles_labels()


def meta_train_uncertainty(
                       fig,
                       subplot_id,
                       env_name,
                       env_name_lc,
                       ncol=1,
                       save_dir="./output/presentation",
                       task_name="Testing_Reward",
                       alpha=0.1,
                       xlim=[0, 1.e6],
                       aggre_step=5,
                       fig_key="AverageReturn_all_test_tasks",
                       last_plot=False,
                       xlog=False,
                       **kwargs):
    print(list(kwargs.values())[0][0])
    if env_name == "Ant-Fwd-Back" or env_name == "Half-Cheetah-Fwd-Back":
        fig_key = "AverageReturn_all_test_tasks0"
    ax = fig.add_subplot(subplot_id)

    for key, dir_lst in kwargs.items():
        y_lst = []
        print('dir_lst', dir_lst)
        min_dim = 99999
        if len(dir_lst) > 1:

            for dir in dir_lst:
                print(dir)
                df = pd.read_csv(dir)
                x = df['Number of env steps total'].to_numpy()
                if x.shape[0] < min_dim:
                    min_dim = x.shape[0]
                y = df[fig_key].to_numpy().reshape((1, -1))
                y_lst.append(y)
            for i, _ in enumerate(y_lst):
                y_lst[i] = y_lst[i][:, :min_dim]
            x = x[:min_dim]
            if len(y_lst) == 0:  # no data, skip plotting
                continue

            y = np.concatenate(y_lst, axis=0)
        else:
            df = pd.read_csv(dir_lst[0])
            x = df['Number of env steps total'].to_numpy()
            if x.shape[0] < min_dim:
                min_dim = x.shape[0]
            y = df[fig_key].to_numpy().reshape((1, -1))

        if aggre_step > 1:
            
            dim1 = int(int((y.shape[1]-1)/aggre_step) * aggre_step) # effective dimension of axis 1 for aggregate plotting
            dim0 = y.shape[0]
            y_data = y[:, 1:1+dim1] # do not average the initial value of x and y
            y_data = y_data.reshape((dim0, -1, int(aggre_step)))
            y_data = y_data.transpose((0, 2, 1)).reshape((dim0 * aggre_step, -1))
            y = np.concatenate([y[: 0], y])
            x_data = x[1:1+dim1]
            x_data = x_data.reshape((-1, int(aggre_step)))
            print(x_data, x_data.shape)
            x_data_mean = np.mean(x_data, axis=1)
            y_data_mean = np.mean(y_data, axis=0)
            y_data_std = np.std(y_data, axis=0)

            print(x[0:1].shape, x_data_mean.shape)
            x = np.concatenate([x[0:1], x_data_mean], axis=0)
            y_mean = np.concatenate([np.mean(y[:, 0:1], axis=0), y_data_mean], axis=0)
            y_std = np.concatenate([np.std(y[:, 0:1], axis=0), y_data_std], axis=0)
            print(y.shape, x.shape)

        else:

            y_mean = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
        
        if key == "SparsePointRobot_Deterministic_LatentSize5":
            key = "Determinisitc"

        elif key == "SparsePointRobot_Stochastic_LatentSize5":
            key = "Stochastic"
       
        x -= x[0]
        l, = ax.plot(x, y_mean, label=key, linewidth=1)
        color = l.get_color()
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=alpha, color=color)
        ax.set_title(env_name, fontsize=17)
        ax.set_xlim(xlim)
        if xlog:
            print('xlog')
            ax.set_xscale('log')
        else:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.set_xlabel("Sample Steps", fontsize=15)
        ax.set_ylabel("Average Return", fontsize=15)

    ax.grid(linestyle='--', linewidth=0.8)
    key_lst = ['context_bcq', 'full_model']
    label_lst = ['ContextualBCQ', 'DistilledBCQ']
    for key, label in zip(key_lst, label_lst):
        npz_lst = glob.glob(f"./output/baseline/evaluation_result/{key}/{env_name_lc}_seed*_eval_result.npz")
        if len(npz_lst) > 0:
            if env_name != 'Sparse-Point-Robot' and env_name != 'Walker-2D-Params':
                plot_baseline(data_dir_lst=npz_lst, ax=ax, alpha=alpha, label=label, xlim=xlim, xlog=xlog)

    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    if last_plot:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=ncol, prop={'size': 15})


def meta_train_uncertainty_multiplot(
               data_dir_lst,
               save_dir="./output/figures",
               env_name_lst=["Sparse-Point-Robot", "Half-Cheetah-Vel", "Ant-Fwd-Back", "Half-Cheetah-Fwd-Back"],
               env_name_lc_lst=["sparse-point-robot", "halfcheetah-vel", "ant-dir", "halfcheetah-dir"],
               subplot_id_lst=[141, 142, 143, 144],
               figsize=(30, 6),
               xlim=[0, 1.e6],
               legend_col=None,
               fig_key="AverageReturn_all_test_tasks",
               aggre_step=5,
               xlog=False,
               save=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    last_plot = False
    if legend_col is None:
        legend_col = len(env_name_lst)
    for data_dir, env_name, env_name_lc, subplot_id in zip(data_dir_lst, env_name_lst, env_name_lc_lst, subplot_id_lst):
        if env_name == env_name_lst[-1]:
            last_plot = True
        meta_train_uncertainty(
            save_dir=save_dir,
            fig=fig,
            FOCAL=data_dir[0],
            BatchPEARL=data_dir[1],
            fig_key=fig_key,
            subplot_id=subplot_id,
            env_name=env_name,
            env_name_lc=env_name_lc,
            xlim=xlim,
            alpha=0.1,
            last_plot=last_plot,
            ncol=legend_col,
            aggre_step=aggre_step,
            xlog=xlog)
    if save:
        fig.savefig(os.path.join(save_dir, 'testing_reward_xlog%s.png' %(xlog)), dpi=200)


def task_dist_2d(df_dir_lst,
                 subplot_id_lst = [221, 224, 222, 223],
                 color_lst = ['r', 'g', 'b', 'm'],
                 subplot_title_lst= ["Inverse-square", "Square"],
                 n_tasks=80,
                 subplot=False):

    fig = plt.figure(figsize=(8, 4.5))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    for df_dir, subplot_id, subplot_title, color in zip(df_dir_lst, subplot_id_lst, subplot_title_lst, color_lst):
        df = pd.read_csv(df_dir)
        z_lst = []
        for task_idx in range(n_tasks):
            z_lst.append(df[df['task_idx'] == task_idx].to_numpy())

        if subplot:
            ax = fig.add_subplot(subplot_id)
            ax.scatter(x, y, s=1)
            ax.set_title(subplot_title)
            ax.set_xlabel('x', fontsize=15)
            ax.set_ylabel('y', fontsize=15)
            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])
        else:
            plt.scatter(df[df['task_idx'] == 0].to_numpy()[:, 2], df[df['task_idx'] == 0].to_numpy()[:, 3],  facecolors='none', edgecolors=color, marker='o', alpha=0.3, label=subplot_title)
            plt.scatter(df[df['task_idx'] == 5].to_numpy()[:, 2], df[df['task_idx'] == 5].to_numpy()[:, 3],  facecolors='none', edgecolors=color, marker='^', alpha=0.3)
            plt.scatter(df[df['task_idx'] == 20].to_numpy()[:, 2], df[df['task_idx'] == 20].to_numpy()[:, 3],  facecolors='none', edgecolors=color, marker='s', alpha=0.3)
            plt.scatter(df[df['task_idx'] == 30].to_numpy()[:, 2], df[df['task_idx'] == 30].to_numpy()[:, 3],  facecolors='none', edgecolors=color, marker='D', alpha=0.3)
    if not subplot:
        plt.legend()
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
    fig.savefig(f"/Users/lanqingli/Documents/Tencent/Agriculture+AI/Technical/Tencent/AgriGroup/Material/working_papers/FOCAL/figures/context_embedding_ntask{n_tasks}.png", dpi=200)

def task_dist_avg(df_dir_lst=["./output/cheetah-vel/9.49.52.138/data_epoch/2020_09_04_23_09_22_seed0/data_epoch_499.csv",
                             "./output/cheetah-vel/9.49.52.138/data_epoch/2020_09_04_23_16_21_seed0/data_epoch_499.csv",
                             "./output/cheetah-vel/9.49.52.138/data_epoch/2020_09_07_14_31_01_seed0/data_epoch_499.csv",
                             "./output/cheetah-vel/9.49.52.138/data_epoch/2020_09_07_14_34_46_seed0/data_epoch_499.csv"],
                  subplot_title_lst= ["Inverse-square", "Square", "Inverse", "Linear"],
                  n_tasks=80):
    for df_dir, title in zip(df_dir_lst, subplot_title_lst):
        df = pd.read_csv(df_dir)
        z_lst = []
        for task_idx in range(n_tasks):
            z_lst.append(np.mean(df[df['task_idx'] == task_idx].to_numpy(), axis=0))
        z_npy = np.array(z_lst)
        print('%s_var' %(title), np.std(z_npy, axis=0))
        print('%s_var_avg' %(title), np.mean(np.std(z_npy, axis=0)[2:7]))
        print('%s_distance_avg' %(title), np.mean(np.std(z_npy, axis=0)[2:7]) * np.sqrt(2 * n_tasks/(n_tasks-1)))

        cnt = 0
        total_cnt = 0
        for j in range(z_npy.shape[0]):
            for k in range(j+1, z_npy.shape[0]):
                    if np.mean((z_npy[j][2:7] - z_npy[k][2:7]) ** 2) > 2/3:
                        cnt += 1
        print('# effective separation pairs: %d' %cnt)
        print('percentage: %f' %(cnt/(n_tasks * (n_tasks - 1))))

def task_dist_2d(
                    save_dir,
                    df_dir_lst,
                    rows=4,
                    cols=2,
                    n_figs=7,
                    color_lst = ['r', 'g', 'b', 'c', 'm', 'y', '#bcbd22', '#17becf'],
                    subplot_title_lst = ["FOCAL_850000_prob_dml", "FOCAL_850000_det_dml", "FOCAL_850000_prob_bellman", "BatchPEARL_850000", "FOCAL_all_prob_dml", "FOCAL_all_det_dml", "FOCAL_all_prob_bellman"],
                    n_tasks = 20,
                    subplot = True,
                    seperate_tasks=True,
                    figsize=None,
                    fontsize=10,
                    method=PCA,
                    method_key='pca',
                    n_points=1600,
                    xlim=None,
                    ylim=None,
                    epoch=10,
                    save=False):
    if figsize is None:
        fig = plt.figure(figsize=(8, 10))
    else:
        fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    if subplot:
        for df_dir, n_fig, subplot_title in zip(df_dir_lst, range(1, n_figs+1), subplot_title_lst):
            df = pd.read_csv(df_dir)[:n_points]
            z = df.to_numpy()[:, 2:7] # latent dimensions

            ax = fig.add_subplot(rows, cols, n_fig)
            proj = projection(z, method)
            print('proj', proj)
            if seperate_tasks:
                for task_idx in range(n_tasks):
                    idxs = df.index[df['task_idx'] == task_idx]
                    print(idxs)
                    ax.scatter(proj[idxs, 0], proj[idxs, 1], s=1, alpha=0.3, cmap=plt.cm.Spectral)
            else:
                ax.scatter(proj[:, 0], proj[:, 1], s=1, alpha=0.1, cmap=plt.cm.Spectral)

            ax.set_title(subplot_title, fontsize=fontsize)
            ax.set_xlabel('x', fontsize=fontsize)
            ax.set_ylabel('y', fontsize=fontsize)
      
    else:
        for df_dir, n_fig, subplot_title, color in zip(df_dir_lst, range(1, n_figs+1), subplot_title_lst, color_lst):
            df = pd.read_csv(df_dir)[:n_points]
            z = df.to_numpy()[:n_points, 2:7] # latent dimensions
            plt.scatter(df[df['task_idx'] == 0].to_numpy()[:, 2], df[df['task_idx'] == 0].to_numpy()[:, 3], facecolors='none',
                        edgecolors=color, marker='o', alpha=0.3, label=subplot_title)
            plt.scatter(df[df['task_idx'] == 5].to_numpy()[:, 2], df[df['task_idx'] == 5].to_numpy()[:, 3], facecolors='none',
                        edgecolors=color, marker='^', alpha=0.3)
            plt.scatter(df[df['task_idx'] == 20].to_numpy()[:, 2], df[df['task_idx'] == 20].to_numpy()[:, 3], facecolors='none',
                        edgecolors=color, marker='s', alpha=0.3)
            plt.scatter(df[df['task_idx'] == 30].to_numpy()[:, 2], df[df['task_idx'] == 30].to_numpy()[:, 3], facecolors='none',
                        edgecolors=color, marker='D', alpha=0.3)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if not subplot:
        plt.legend()
    env_name = df_dir_lst[0].split("/")[2]

    os.makedirs(save_dir, exist_ok=True)

    if save:
        if seperate_tasks:
            fig.savefig(
                os.path.join(save_dir, f"{env_name}_context_embedding_{method_key}_ntask{n_tasks}_separatetasks_epoch{epoch}.png"),
                dpi=200)
        else:
            fig.savefig(
                os.path.join(save_dir, f"{env_name}_context_embedding_{method_key}_ntask{n_tasks}_epoch{epoch}.png"),
                dpi=200)
        plt.close()

def projection(z, method):
    print(z.shape)
    proj = method(n_components=2).fit_transform(X=z)
    return proj
               
def figure2a(save_dir="./output/figures"):
    ###########################################################################################
    # Reproducing Fig 2a: Context Embeddings for four choices of DML losses on Half-Cheetah-Vel  
    ###########################################################################################
    task_dist_2d(
                df_dir_lst=["./output/cheetah-vel/2020_09_04_23_09_22_seed0/data_epoch_499.csv",
                            "./output/cheetah-vel/2020_09_07_14_31_01_seed0/data_epoch_499.csv",
                            "./output/cheetah-vel/2020_09_07_14_34_46_seed0/data_epoch_499.csv",
                            "./output/cheetah-vel/2020_09_04_23_16_21_seed0/data_epoch_499.csv"],
                rows=2,
                cols=2,
                n_figs=4,
                color_lst=['r', 'g', 'b', 'm'],
                subplot_title_lst= ["Inverse-square", "Inverse", "Linear", "Square"],
                n_tasks=20,
                subplot=True,
                seperate_tasks=True,
                figsize=[10, 8],
                save_dir="./output/figures/",
                fontsize=15,
                n_points=16000)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "figure2a.png"), dpi=200)
    plt.close()


def figure2b(save_dir="./output/figures"):
    ###########################################################################################
    # Reproducing Fig 2b: Return Curves for four choices of DML losses on Half-Cheetah-Vel  
    ###########################################################################################
    data_dir_lst = [
           ["./output/cheetah-vel/2020_09_04_23_09_22_seed0/progress.csv",
            "./output/cheetah-vel/2020_09_04_23_09_22_seed2/progress.csv",
            "./output/cheetah-vel/2020_09_04_23_09_19_seed1/progress.csv",
           ],
           ["./output/cheetah-vel/2020_09_07_14_31_01_seed0/progress.csv",
            "./output/cheetah-vel/2020_09_07_14_31_01_seed1/progress.csv",
            "./output/cheetah-vel/2020_09_07_14_31_01_seed2/progress.csv",
           ],
           ["./output/cheetah-vel/2020_09_07_14_34_46_seed0/progress.csv",
            "./output/cheetah-vel/2020_09_07_14_34_46_seed1/progress.csv",
            "./output/cheetah-vel/2020_09_07_14_34_46_seed2/progress.csv",
           ],
           ["./output/cheetah-vel/2020_09_04_23_16_21_seed0/progress.csv",
            "./output/cheetah-vel/2020_09_04_23_16_21_seed2/progress.csv",
            "./output/cheetah-vel/2020_09_04_23_16_25_seed1/progress.csv",
           ]]
    env_name_lst = ["Inverse-square", "Inverse", "Linear", "Square"]
    return_curve(
        data_dir_lst=data_dir_lst,
        env_name_lst=env_name_lst,
        aggre_step=5,
        title="Half-Cheetah-Vel",
        xlim=[0, 2.7e6],
        fig_keys=["AverageReturn_all_test_tasks"]
    ) 
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "figure2b.png"), dpi=200)
    plt.close()


def figure3(xlog=False,
            save_dir="./output/figures"):    
    ##################################################################################
    # Reproducing Fig 3: FOCAL vs. Baselines (Batch PEARL, ContextualBCQ, DistilledBCQ)
    ##################################################################################
    data_dir_lst = [[
        [
            "./output/sparse-point-robot/2020_11_21_23_49_57_seed1/progress.csv",
            "./output/sparse-point-robot/2020_11_21_23_49_57_seed2/progress.csv",
            "./output/sparse-point-robot/2020_11_21_23_49_58_seed0/progress.csv",
        ],
        [
            "./output/sparse-point-robot/2020_11_22_00_02_50_seed0/progress.csv",
            "./output/sparse-point-robot/2020_11_22_00_02_50_seed1/progress.csv",
            "./output/sparse-point-robot/2020_11_22_00_02_50_seed2/progress.csv",
        ]],
        [[
            "./output/cheetah-vel/2020_11_24_18_17_06_seed1/progress.csv",
            "./output/cheetah-vel/2020_11_24_18_17_06_seed2/progress.csv",
            "./output/cheetah-vel/2020_11_24_18_17_13_seed0/progress.csv",
        ],
        [
             "./output/cheetah-vel/2020_08_16_12_36_05_seed0/progress.csv",
             "./output/cheetah-vel/2020_08_16_12_36_07_seed1/progress.csv",
             "./output/cheetah-vel/2020_08_16_12_36_08_seed2/progress.csv",
        ]],
        [[
    
              "./output/ant-dir/2020_08_13_19_51_11_seed2/progress.csv",
              "./output/ant-dir/2020_08_13_19_51_11_seed0/progress.csv",
              "./output/ant-dir/2020_08_13_19_51_11_seed1/progress.csv",
        ],
        [
              "./output/ant-dir/2020_08_16_12_43_08_seed1/progress.csv",
              "./output/ant-dir/2020_08_16_12_43_09_seed0/progress.csv",
              "./output/ant-dir/2020_08_16_12_43_09_seed2/progress.csv",
        ]],
        [[
            "./output/cheetah-dir/2020_08_07_14_27_50_seed1/progress.csv",
            "./output/cheetah-dir/2020_08_07_14_27_50_seed2/progress.csv",
            "./output/cheetah-dir/2020_08_07_14_27_53_seed0/progress.csv",
        ],
        [
            "./output/cheetah-dir/2020_08_24_23_52_22_seed0/progress.csv",
            "./output/cheetah-dir/2020_08_24_23_52_22_seed1/progress.csv",
            "./output/cheetah-dir/2020_08_24_23_52_22_seed2/progress.csv",
        ]]]
    meta_train_uncertainty_multiplot(
               data_dir_lst=data_dir_lst,
               save_dir="./output/figures",
               env_name_lst=["Sparse-Point-Robot", "Half-Cheetah-Vel", "Ant-Fwd-Back", "Half-Cheetah-Fwd-Back"],
               env_name_lc_lst=["sparse-point-robot", "halfcheetah-vel", "ant-dir", "halfcheetah-dir"],
               subplot_id_lst=[141, 142, 143, 144],
               figsize=(30, 6),
               xlim=[0, 1.e6],
               legend_col=None,
               fig_key="AverageReturn_all_test_tasks",
               aggre_step=5,
               xlog=xlog
               )
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "figure3.png"), dpi=200)
    plt.close()

def figure4a(save_dir="./output/figures"):
    #######################################################################################################
    # Reproducing Fig 4a: Context Embeddings for two design choices of context encoders on Walker-2D-Params  
    #######################################################################################################
    # walker-rand-params
    task_dist_2d(
                    df_dir_lst = [
                                  "./output/walker-rand-params/2020_09_30_22_14_23_seed0/data_epoch_10.csv",
                                  "./output/walker-rand-params/2020_09_30_22_12_01_seed0/data_epoch_10.csv",
                                  ],
                    rows=1,
                    cols=2,
                    n_figs=2,
                    color_lst = ['r', 'g', 'b', 'm'],
                    subplot_title_lst = ["FOCAL (Deterministic)", "FOCAL (Probabilistic)"], 
                    n_tasks = 20,
                    subplot = True,
                    seperate_tasks=True,
                    figsize=[10, 4.5],
                    save_dir="./output/figures/",
                    fontsize=15,
                    n_points=16000) 
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "figure4a.png"), dpi=200)
    plt.close()

def figure4b(save_dir="./output/figures"):
    ####################################################################################################
    # Reproducing Fig 4b: Return Curves for two design choices of context encoders on Sparse-Point-Robot  
    ####################################################################################################
    data_dir_lst = [
           ["./output/sparse-point-robot/2020_07_22_00_16_46_seed0/progress.csv",
            "./output/sparse-point-robot/2020_07_22_00_16_46_seed1/progress.csv",
            "./output/sparse-point-robot/2020_07_22_00_16_46_seed2/progress.csv"],
           ["./output/sparse-point-robot/2020_07_08_14_35_40_seed0/progress.csv",
            "./output/sparse-point-robot/2020_07_08_14_35_40_seed1/progress.csv",
            "./output/sparse-point-robot/2020_07_08_14_35_41_seed2/progress.csv"]]
    env_name_lst = ["FOCAL (Deterministic)", "FOCAL (Probabilistic)"]
    return_curve(
        data_dir_lst=data_dir_lst,
        env_name_lst=env_name_lst,
        env_name="sparse-point-robot",
        title="Sparse-Point-Robot",
        xlim=[0, 1000000],
        fig_keys=["AverageReturn_all_test_tasks"]
    )
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "figure4b.png"), dpi=200)
    plt.close()

def figure5a(save_dir="./output/figures"):
    #######################################################################################################
    # Reproducing Fig 5a: Context Embeddings for two design choices of training strategy on Walker-2D-Params  
    #######################################################################################################
    task_dist_2d(
            df_dir_lst=[
            "./output/walker-rand-params/2020_11_21_10_00_12_seed2/data_epoch_50.csv",
            "./output/walker-rand-params/2020_11_21_09_46_55_seed2/data_epoch_50.csv",
            ],
            rows=1, 
            cols=2,
            n_figs=2,
            figsize=[10,4.5],
            subplot_title_lst= ["FOCAL (ours)", "FOCAL (Bellman)"],
            save_dir="./output/figures/",
            fontsize=15,
            n_points=16000)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "figure5a.png"), dpi=200)
    plt.close()

def figure5b(save_dir="./output/figures"):
    ###################################################################################################
    # Reproducing Fig 5b: Return Curves for two design choices of training strategy on Walker-2D-Params  
    ###################################################################################################
    data_dir_lst = [
            [
            "./output/walker-rand-params/2020_11_21_10_00_08_seed0/progress.csv",
            "./output/walker-rand-params/2020_11_21_10_00_12_seed1/progress.csv",
            "./output/walker-rand-params/2020_11_21_10_00_12_seed2/progress.csv"],
           ["./output/walker-rand-params/2020_11_21_09_46_51_seed0/progress.csv",
            "./output/walker-rand-params/2020_11_21_09_46_55_seed2/progress.csv",
            "./output/walker-rand-params/2020_11_21_09_46_57_seed1/progress.csv"]]
    env_name_lst = ["FOCAL (ours)", "FOCAL (Bellman)"]
    return_curve(
        data_dir_lst=data_dir_lst,
        env_name_lst=env_name_lst,
        env_name="walker-rand-params",
        title="Walker-2D-Params",
        fig_keys=["AverageReturn_all_test_tasks"],
        xlim=[0, 1.3e6],
        aggre_step=1)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "figure5b.png"), dpi=200)
    plt.close()

def figure6(xlog=True,
            save_dir="./output/figures"):
     ##################################################################################
    # Reproducing Fig 6: FOCAL vs. Baselines (Batch PEARL, ContextualBCQ, DistilledBCQ)
    ##################################################################################
    data_dir_lst = [[
        [
            "./output/sparse-point-robot/2020_11_21_23_49_57_seed1/progress.csv",
            "./output/sparse-point-robot/2020_11_21_23_49_57_seed2/progress.csv",
            "./output/sparse-point-robot/2020_11_21_23_49_58_seed0/progress.csv",
        ],
        [
            "./output/sparse-point-robot/2020_11_22_00_02_50_seed0/progress.csv",
            "./output/sparse-point-robot/2020_11_22_00_02_50_seed1/progress.csv",
            "./output/sparse-point-robot/2020_11_22_00_02_50_seed2/progress.csv",
        ]],
        [[
            "./output/walker-rand-params/2020_11_21_10_00_08_seed0/progress.csv",
            "./output/walker-rand-params/2020_11_21_10_00_12_seed1/progress.csv",
            "./output/walker-rand-params/2020_11_21_10_00_12_seed2/progress.csv",
        ],
        [
            "./output/walker-rand-params/2020_11_22_12_29_03_seed1/progress.csv",
            "./output/walker-rand-params/2020_11_22_12_28_58_seed2/progress.csv",
            "./output/walker-rand-params/2020_11_22_12_29_07_seed0/progress.csv",
        ]],
        [[
            "./output/cheetah-vel/2020_11_24_18_17_06_seed1/progress.csv",
            "./output/cheetah-vel/2020_11_24_18_17_06_seed2/progress.csv",
            "./output/cheetah-vel/2020_11_24_18_17_13_seed0/progress.csv",
        ],
        [
             "./output/cheetah-vel/2020_08_16_12_36_05_seed0/progress.csv",
             "./output/cheetah-vel/2020_08_16_12_36_07_seed1/progress.csv",
             "./output/cheetah-vel/2020_08_16_12_36_08_seed2/progress.csv",
        ]],
        [[
    
              "./output/ant-dir/2020_08_13_19_51_11_seed2/progress.csv",
              "./output/ant-dir/2020_08_13_19_51_11_seed0/progress.csv",
              "./output/ant-dir/2020_08_13_19_51_11_seed1/progress.csv",
        ],
        [
              "./output/ant-dir/2020_08_16_12_43_08_seed1/progress.csv",
              "./output/ant-dir/2020_08_16_12_43_09_seed0/progress.csv",
              "./output/ant-dir/2020_08_16_12_43_09_seed2/progress.csv",
        ]],
        [[
            "./output/cheetah-dir/2020_08_07_14_27_50_seed1/progress.csv",
            "./output/cheetah-dir/2020_08_07_14_27_50_seed2/progress.csv",
            "./output/cheetah-dir/2020_08_07_14_27_53_seed0/progress.csv",
        ],
        [
            "./output/cheetah-dir/2020_08_24_23_52_22_seed0/progress.csv",
            "./output/cheetah-dir/2020_08_24_23_52_22_seed1/progress.csv",
            "./output/cheetah-dir/2020_08_24_23_52_22_seed2/progress.csv",
        ]]]
    meta_train_uncertainty_multiplot(
               data_dir_lst=data_dir_lst,
               save_dir="./output/figures",
               env_name_lst=["Sparse-Point-Robot", "Walker-2D-Params", "Half-Cheetah-Vel", "Ant-Fwd-Back", "Half-Cheetah-Fwd-Back"],
               env_name_lc_lst=["sparse-point-robot", "walker-rand-params", "halfcheetah-vel", "ant-dir", "halfcheetah-dir"],
               subplot_id_lst=[231, 233, 234, 235, 236],
               figsize=(15, 10),
               xlim=[0, 5.e6],
               legend_col=None,
               fig_key="AverageReturn_all_test_tasks",
               aggre_step=5,
               xlog=xlog)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "figure6.png"), dpi=200)
    plt.close()

if __name__ == '__main__':
    fig = plt.figure()
    figure2a()
    figure2b()
    figure3()
    figure4a()
    figure4b()
    figure5a()
    figure5b()
    figure6()
   