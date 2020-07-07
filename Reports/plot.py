import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True, font_scale=1)
import itertools
# marker = itertools.cycle(("o", "^", "s")) 
marker = itertools.cycle((None,)) 
markevery = 20
from tensorboard.backend.event_processing import event_accumulator

def plot_result_csv(folders, filenames, files, max_length = 800, ifsave = False, sns = sns, marker = marker, markerevery = markevery):
    # input should like this:
    #
    # folders = ['new_section_baseline', 'pb_c_base_500','pb_c_base_100','very_different_exploration']
    # filenames = ['Total reward']
    # files = [[['Tensorboard_data/{}/1/{}.csv'.format(folder, filename),
    #         'Tensorboard_data/{}/2/{}.csv'.format(folder, filename),
    #         'Tensorboard_data/{}/3/{}.csv'.format(folder, filename),]
    #         for filename in filenames] for folder in folders]
    data = np.array([[[np.loadtxt(tb_file, delimiter=',', skiprows=1)[:max_length,2] 
                    for tb_file in index_file]
                    for index_file in folder_file]
                    for folder_file in files])
    data_mean = np.mean(data, axis=2)
    data_std = np.std(data, axis=2)
    
    clrs = sns.color_palette(n_colors=len(folders))

    epochs = list(range(max_length))
    for j in range(len(filenames)):
        fig, ax = plt.subplots()
        for i in range(len(folders)):
            if len(epochs) > len(data_mean[i,j]):
                epochs = list(range(len(data_mean[i,j])))
            ax.plot(epochs, data_mean[i][j], label=folders[i], marker=next(marker), markevery=markevery, c=clrs[i])
            ax.fill_between(epochs, data_mean[i][j]-data_std[i][j], data_mean[i][j]+data_std[i][j] ,alpha=0.3, facecolor=clrs[i])
        ax.set_xlabel('Training iteration', color='k')  
        ax.set_ylabel(filenames[j], color='k') 
        ax.legend()
        plt.show()
        if ifsave:
            fig.savefig('{}.png'.format(filenames[j]), format='png', dpi=600)

def plot_result_event(folders, filenames, files, max_length = 800, ifsave = False, labels = [], sns = sns, marker = marker, markerevery = markevery):
    # input should like this:
    #
    # folders = ['new_section_baseline', 'pb_c_base_500','pb_c_base_100']
    # filenames = ['1.Reward/1.Total reward']
    # labels = ['Total reward']
    # files = [['Tensorboard_data/{}/1'.format(folder),
    #         'Tensorboard_data/{}/2'.format(folder),
    #         'Tensorboard_data/{}/3'.format(folder),]
    #         for folder in folders]
    data = []
    for i in range(len(files)):
        repeated_experiments = []
        for j in range(len(files[0])):
            event = event_accumulator.EventAccumulator(files[i][j])
            event.Reload()
            diff_index = []
            for k in range(len(filenames)):
                try:
                    scalars = event.scalars.Items(filenames[k])
                except:
                    print(event.scalars.Keys())
                    return None
                diff_index.append([item.value for item in scalars][:max_length])   
            repeated_experiments.append(diff_index)
        data.append(repeated_experiments)
        
        
    data = np.array(data)
    
    data = np.transpose(data, (0,2,1,3))
    data_mean = np.mean(data, axis=2)
    data_std = np.std(data, axis=2)
    
    clrs = sns.color_palette(n_colors=len(folders))

    if max_length > len(data_mean[0][0]):
        scope = len(data_mean[0][0])
    else:
        scope = max_length

    epochs = list(range(scope))
    for j in range(len(filenames)):
        fig, ax = plt.subplots()
        for i in range(len(folders)):
            ax.plot(epochs, data_mean[i][j][:scope], label=folders[i], marker=next(marker), markevery=markevery, c=clrs[i])
            ax.fill_between(epochs, data_mean[i][j][:scope]-data_std[i][j][:scope], data_mean[i][j][:scope]+data_std[i][j][:scope] ,alpha=0.3, facecolor=clrs[i])
        ax.set_xlabel('Training iteration', color='k')  
        if labels == []:
            ax.set_ylabel(filenames[j], color='k') 
        else:
            ax.set_ylabel(labels[j], color='k') 
        ax.legend()
        plt.show()
        if ifsave:
            fig.savefig('{}.png'.format(filenames[j]), format='png', dpi=600)