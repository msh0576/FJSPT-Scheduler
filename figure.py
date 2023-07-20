from operator import is_
import torch
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import os

FONT_SIZE = 5
FIGSIZE = (5.5,5)
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 13
BLUE_SERIES = ['#0000ff', '#0066ff', '#3399ff', '#66ccff']
RED_SERIES = ['#cc0000', '#ff5050', '#ff9999' , '#ffcccc']
GREEN_SERIES = ['#006600', '#00cc00', '#00ff00', '#66ff99']

def set_figure_size(ax, xticks=None, yticks=None, title='', xlabel='', ylabel=''):
    # === title, label size and bole ===
    ax.set_title(title, size=BIGGER_SIZE)
    ax.set_xlabel(xlabel, size=BIGGER_SIZE, weight='bold')
    ax.set_ylabel(ylabel, size=BIGGER_SIZE, weight='bold')

    # === x,y ticks size and weight ===
    for tick in ax.xaxis.get_major_ticks():     
        tick.label1.set_fontsize(MEDIUM_SIZE)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(MEDIUM_SIZE)
        tick.label1.set_fontweight('bold')

    # === x and y ticks ===
    if xticks != None:
        ax.xaxis.set_ticks(xticks)
        ax.set_xlim(min(xticks)-1, max(xticks)+1)
    if yticks != None:
        ax.yaxis.set_ticks(yticks)
        ax.set_ylim(min(yticks), max(yticks))


def plot_figure_setting():
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def get_file_contents(paths, algorithms, labels, result="Score:"):
    '''
    Input:
        paths = [log_file1, log_file2, ...]
        algorithms = [algorithm1, algorithm2, ...]
        labels = [label1, label2, ...]
    Output:
        algorithm_results: {
            "algorithm_name": [list of contents],
            "algorithm_name": [list of contents]
        }
        
    '''
    # paths = file_dict['paths']
    # algorithms = file_dict['algorithms']
    # === record file contests wrt algorithms ===
    algorithm_results = {label:[] for label in labels}
    for path in paths:    
        with open(path, 'r') as file:
            lines = file.readlines()
            for idx, algorithm in enumerate(algorithms):
                does_exist = False
                for line in lines:
                    line_split = line.split()
                    if algorithm in line_split:
                        does_exist = True
                        if result == 'Score:':
                            makespan_idx = line_split.index(result)+1
                            algorithm_results[labels[idx]].append(float(line_split[makespan_idx]))
                        elif result == 'SpandTime:':
                            spandtime_dix = line_split.index(result)+1
                            algorithm_results[labels[idx]].append(float(line_split[spandtime_dix]))
                        else:
                            raise Exception("result input error!")
                if not does_exist:
                    raise Exception( "{} does not exist in {}".format(algorithm, path))
    return algorithm_results
    

def show_result_wrt_graph_size(
    file_dict, result='Score:', xlabel=None, ylabel=None, title=None,
    graph_style='bar',
    same_algo_multi_lines=False,
    save_path=None, save_file_format=None,
):
    '''
    Input:
        result: "Score:" or "SpandTime:"
        graph_style: 'bar' or 'normal'
        figures: (fig, ax)
    '''
    fig, ax = plt.subplots()
    
    paths = file_dict['paths']
    algorithms = file_dict['algorithms']
    labels = file_dict['labels']
    xticks = file_dict['xticks']
    # xticklabels = file_dict['xticklabels']
    line_style = file_dict['line_style']
    # === record file contests wrt algorithms ===
    
    
    if same_algo_multi_lines:
        algorithm_results = {}
        for idx in range(len(file_dict['paths'])):
            algorithm_results.update(get_file_contents(paths[idx], [algorithms[idx]], [labels[idx]], result=result))
    else:
        algorithm_results = get_file_contents(paths, algorithms, labels, result=result)
    # print(f'algorithm_results;{algorithm_results}')
        
    dataframes = []
    for idx, algorithm in enumerate(algorithm_results):
        dataframe = {}
        dataframe[ylabel] = []
        dataframe[xlabel] = []
        dataframe['algorithm'] = []
        
        values = algorithm_results[algorithm]
        dataframe[xlabel].extend(xticks)
        dataframe[ylabel].extend(values)
        dataframe['algorithm'].extend([labels[idx]]*len(values))
        dataframes.append(dataframe)
    # print(f'dataframe:{dataframe}')
    
    dfs = []
    for dataframe_ in dataframes:
        df = pd.DataFrame.from_dict(dataframe_)
        dfs.append(df)
        # print(f'df:{df}')
    total_df = pd.concat(dfs, axis=0)
    # === show bar results ===
    if graph_style == 'bar':
        sns.barplot(x=xlabel, y=ylabel, hue='algorithm', data=total_df)
        set_figure_size(ax=ax, xlabel=xlabel, ylabel=ylabel)
    elif graph_style == 'normal':
        for df_ in dfs:
            ax = df_.plot(kind='line', x=xlabel, y=ylabel, ax=ax, style=line_style)
        plt.legend(labels)
        set_figure_size(ax=ax, xlabel=xlabel, ylabel=ylabel, xticks=xticks)
    else:
        raise Exception('graph style error!')
    
    if save_path is not None:
        if save_file_format is not None:
            plt.savefig(save_path+'.eps', bbox_inches='tight',pad_inches = 0.02, format='eps')
        else:
            plt.savefig(save_path, bbox_inches='tight',pad_inches = 0.02)
    else:
        plt.show()
        
def print_gap(
    file_dict, is_avg_gap=False
):
    paths = file_dict['paths']
    algorithms = file_dict['algorithms']
    labels = file_dict['labels']
    xticks = file_dict['xticks']
    line_style = file_dict['line_style']

    # === record file contests wrt algorithms ===
    algorithm_results_score = get_file_contents(paths, algorithms, labels, result='Score:')
    algorithm_results_runtime = get_file_contents(paths, algorithms, labels, result='SpandTime:')
    
    # print(f'algorithm_results:{algorithm_results_score}')
    
    # === represent score corresponding to instance ===
    score_dict = {}
    for inst_idx, instance in enumerate(xticks):
        score_dict[instance] = {}
        for algorithm, values in algorithm_results_score.items():
            score_dict[instance][algorithm] = values[inst_idx]
    # === represent runtime corresponding to instance ===
    runtime_dict = {}
    for inst_idx, instance in enumerate(xticks):
        runtime_dict[instance] = {}
        for algorithm, values in algorithm_results_runtime.items():
            runtime_dict[instance][algorithm] = values[inst_idx]
    # === compute gap ===
    gap_dict = {}
    for inst_idx, instance in enumerate(xticks):
        gap_dict[instance] = {}
        min_score = 10000000000
        # fine min_score
        for algorithm, values in algorithm_results_score.items():
            if min_score > values[inst_idx]:
                min_score = values[inst_idx]
        # compute gap
        for algorithm, values in algorithm_results_score.items():
            gap = round((values[inst_idx] - min_score)/ min_score, 2)
            gap_dict[instance][algorithm] = gap
    
    
    
    # === print ===
    print(f'===========================')
    print(f'[Score results]')
    for instance, result in score_dict.items():
        print(f'{instance} - {result}')
    print(f'===========================')
    print(f'[Gap results]')
    for instance, result in gap_dict.items():
        print(f'{instance} - {result}')
    print(f'===========================')
    print(f'[Runtime results]')
    for instance, result in runtime_dict.items():
        print(f'{instance} - {result}')

    # === compute average gap ===
    if is_avg_gap:
        avg_gap = {}
        num_instance = len(gap_dict.keys())
        for algorithm in algorithm_results_score.keys():
            avg_gap[algorithm] = 0
        
        for instance in gap_dict.keys():
            for algorithm in gap_dict[instance].keys():
                avg_gap[algorithm] += gap_dict[instance][algorithm]
        for algorithm in avg_gap.keys():
            avg_gap[algorithm] = round(avg_gap[algorithm]/num_instance, 2)
        print(f'===========================')
        print(f'Average Gap')
        print(f'{avg_gap}')

def figure_relative_gap(
    file_dict, save_path=None
):
    paths = file_dict['paths']
    algorithms = file_dict['algorithms']
    labels = file_dict['labels']
    xticks = file_dict['xticks']
    line_style = file_dict['line_style']
    num_instance = len(xticks)
    
    batch_scores_dict = {}
    for path_idx, path in enumerate(paths):
        data = torch.load(path)
        instance = xticks[path_idx]
        result_log = data['result_log'][1]
        batch_scores_dict[instance] = {}
        for algorithm in algorithms:
            batch_scores_dict[instance][algorithm] = result_log[algorithm+'_batch_scores'][0]
            
        

    for inst_idx, instance in enumerate(xticks):
        fig, ax = plt.subplots()
        
        # min_values = [min(values) for values in zip(*batch_scores_dict[instance].values())]
        
        batch_scores = [values for values in zip(*batch_scores_dict[instance].values())]
        batch_scores = np.array(batch_scores) # [B, num_alg] 
        min_batch_scores = batch_scores.min(axis=1, keepdims=True) # [B,1]
        our_batch_scores = np.expand_dims(batch_scores[:, 0], axis=1)   # [B, 1]
        relat_gap = (batch_scores - our_batch_scores) / our_batch_scores * 100  # [B, num_alg]
        relat_gap_avg = relat_gap.mean(axis=0)
        batch_score_mean = batch_scores.mean(axis=0)
        # relat_gap = round(relat_gap, 2)
        
        # print(f'batch_scores_dict:{batch_scores_dict[instance]}')
        # print(f'batch_scores:{batch_scores}')
        # print(f'relat_gap:{relat_gap}')
        # print(f'batch_score_mean: {batch_score_mean}')
        print(f'{instance}_relat_gap_avg:{relat_gap_avg}')
        
        ax.boxplot(relat_gap, vert=True, patch_artist=True, labels=labels)
        # plt.ylabel('Relatively Gap (%)')
        # set_figure_size(ax=ax, ylabel='Relatively Gap (%)')
        ax.set_ylabel('Relative Gap (%)', size=18, weight='bold')
        
        # === x,y ticks size and weight ===
        for tick in ax.xaxis.get_major_ticks():     
            tick.label1.set_fontsize(14)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(18)
            tick.label1.set_fontweight('bold')
        # === save figure ===
        if save_path is not None:
            save_path_ = os.path.join(save_path, 'analy_HGS_'+instance)
            # plt.savefig(save_path_, bbox_inches='tight',pad_inches = 0.02)
            plt.savefig(save_path_+'.eps', bbox_inches='tight',pad_inches = 0.02, format='eps')
        else:
            plt.show()

        
    
    min_batch_rewards = []
    



if __name__ == "__main__":
    result_path = './result/Final_result/'
    save_path = './figure_results/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # ===== makespan, gap and running time on small-scale instances =====
    file_dict = {
        "paths": [
            result_path+'20230510_211922_test_5_3_3/log.txt',
            result_path+'20230510_211922_test_10_3_6/log.txt',
            result_path+'20230510_211922_test_10_6_3/log.txt',
            result_path+'20230510_211922_test_10_6_6/log.txt',
        ],
        "algorithms":[
            "gtrans_jobcentric_5_3_3_EncV2_DecV5",
            "gtrans_jobcentric_10_3_6_EncV2_DecV5",
            "gtrans_jobcentric_10_6_3_EncV2_DecV5",
            "gtrans_jobcentric_10_6_6_EncV2_DecV5",
            # "dtrans_jobcentric_10_6_6_EncV4_DecV5",
            # "dtrans_jobcentric_10_6_6_EncV6_DecV5",
            # "dtrans_jobcentric_10_6_6_EncV5_DecV5",
            # "dtrans_jobcentric_10_6_6_EncV4_DecV5_sec",
            "matnet_jobcentric_10_6_6",
            "hgnn_jobcentric_10_6_6",
            "dispatch_spt",
            "dispatch_lpt",
            "dispatch_fifo",
            "GA"
        ],
        "labels":[
            "HGS (5X3X3)",
            "HGS (10X3X6)",
            "HGS (10X6X3)",
            "HGS (10X6X6)",
            # "HGS (EncV4)",
            # "HGS (EncV6)",
            # "HGS (EncV5)",
            # "HGS (EncV4_sec)" ,
            "MatNet",
            "HGNN",
            "SPT",
            "LPT",
            "FIFO",
            "IGA"
        ],
        "xticks":[
            '5X3X3', '10X3X6', '10X6X3', '10X6X6'
        ],
        "line_style" : None,
        
    }
    assert len(file_dict['algorithms']) == len(file_dict['labels']) 
    assert len(file_dict['paths']) == len(file_dict['xticks']) 
    print(f'====== small-scale instances ===== ')
    print_gap(file_dict, is_avg_gap=True)
    
    # ===== makespan, gap and running time on large-scale instances =====
    file_dict = {
        "paths": [
            result_path+'20230510_211922_test_20_10_10/log.txt',
            result_path+'20230510_211922_test_30_15_15/log.txt',
            result_path+'20230510_211922_test_40_20_20/log.txt',
            result_path+'20230510_211922_test_50_25_25/log.txt',
        ],
        "algorithms":[
            "gtrans_jobcentric_10_6_6_EncV2_DecV5",
            # "dtrans_jobcentric_10_6_6_EncV4_DecV5",
            # "dtrans_jobcentric_10_6_6_EncV6_DecV5",
            # "dtrans_jobcentric_10_6_6_EncV5_DecV5",
            # "dtrans_jobcentric_10_6_6_EncV4_DecV5_sec",
            "matnet_jobcentric_10_6_6",
            "hgnn_jobcentric_10_6_6",
            "dispatch_spt",
            "dispatch_lpt",
            "dispatch_fifo",
            "GA"
        ],
        "labels":[
            "HGS",
            # "HGS (EncV4)",
            # "HGS (EncV6)",
            # "HGS (EncV5)",
            # "HGS (EncV4_sec)" ,
            "MatNet",
            "HGNN",
            "SPT",
            "LPT",
            "FIFO",
            "IGA"
        ],
        "xticks":[
            '20X10X10', '30X15X15', '40X20X20', '50X25X25'
        ],
        # "xticklabels":[
        #     "20",
        #     "40",
        # ],
        "line_style" : None,
        
    }
    assert len(file_dict['algorithms']) == len(file_dict['labels']) 
    assert len(file_dict['paths']) == len(file_dict['xticks']) 
    print(f'====== large-scale instances ===== ')
    print_gap(file_dict, is_avg_gap=True)
    
    # : graph size = job10_ma5_veh?
    # show_result_wrt_graph_size(file_dict, result='Score:',
    #     xlabel='Graph size', ylabel='Makespan',
    #     graph_style='bar',
    #     # save_path=os.path.join(save_path, 'makespan_wrt_graphsize')
    # )
    
    # show_result_wrt_graph_size(file_dict, result='SpandTime:',
    #     xlabel='Graph size', ylabel='Makespan',
    #     graph_style='bar',
    #     # save_path=os.path.join(save_path, 'makespan_wrt_graphsize')
    # )
    # ===========================================================
    # ======= proposed method analysis : NoEnc, NoDec, self-attention ========
    file_dict = {
        "paths": [
            result_path+'20230510_211922_test_10_6_6/log.txt',
            result_path+'20230510_211922_test_20_10_10/log.txt',
            result_path+'20230510_211922_test_30_15_15/log.txt',
            result_path+'20230510_211922_test_40_20_20/log.txt',
            result_path+'20230510_211922_test_50_25_25/log.txt',
        ],
        "algorithms":[
            "gtrans_jobcentric_10_6_6_EncV2_DecV5",
            # "dtrans_jobcentric_10_6_6_EncV6_DecV5",
            # "dtrans_jobcentric_10_6_6_EncV5_DecV5",
            # "dtrans_jobcentric_10_6_6_EncV0_DecV5",
            "dtrans_jobcentric_10_6_6_EncV5_DecV0",
            "dtrans_jobcentric_10_6_6_EncV1_DecV5", # self-attention, ignoring edge attributes
            # "dtrans_jobcentric_10_6_6_EncV4_DecV5_sec",
        ],
        "labels":[
            "HGS",
            # "HGS (EncV6)",
            # "HGS (EncV5)",
            # "HGS (NoEnc)",
            "HGS (NoDec)",
            "HGS (with self-atten)",
            # "HGS (EncV4_2)",
        ],
        "xticks":[
            '10X6X6', 
            '20X10X10', 
            '30X15X15', 
            '40X20X20', 
            '50X25X25'
        ],
        "line_style" : None,
    }
    assert len(file_dict['algorithms']) == len(file_dict['labels']) 
    assert len(file_dict['paths']) == len(file_dict['xticks']) 
    
    # print(f'====== Proposed method analysis ===== ')
    # print_gap(file_dict, is_avg_gap=True)
    
    
    # ===========================================================
    # ============ benckmark dataset1 ==============
    file_dict = {
        "paths": [
            result_path+'20230510_211736_test_Mk01.fjs/log.txt',
            result_path+'20230510_211736_test_Mk02.fjs/log.txt',
            result_path+'20230510_211736_test_Mk03.fjs/log.txt',
            result_path+'20230510_211736_test_Mk04.fjs/log.txt',
            result_path+'20230510_211736_test_Mk05.fjs/log.txt',
            result_path+'20230510_211736_test_Mk06.fjs/log.txt',
            result_path+'20230510_211736_test_Mk07.fjs/log.txt',
            result_path+'20230510_211736_test_Mk08.fjs/log.txt',
            result_path+'20230510_211736_test_Mk09.fjs/log.txt',
            result_path+'20230510_211736_test_Mk10.fjs/log.txt',
        ],
        "algorithms":[
            "gtrans_jobcentric_10_6_6_EncV2_DecV5",
            # "dtrans_jobcentric_10_6_6_EncV4_DecV5",
            # "dtrans_jobcentric_10_6_6_EncV6_DecV5",
            # "dtrans_jobcentric_10_6_6_EncV5_DecV5",
            # "dtrans_jobcentric_10_6_6_EncV4_DecV5_sec",
            "matnet_jobcentric_10_6_6",
            "hgnn_jobcentric_10_6_6",
            "dispatch_spt",
            "dispatch_lpt",
            "dispatch_fifo",
            "GA"
        ],
        "labels":[
            "HGS",
            # "HGS (EncV4)",
            # "HGS (EncV6)",
            # "HGS (EncV5)",
            # "HGS (EncV4_2)",
            "MatNet",
            "HGNN",
            "SPT",
            "LPT",
            "FIFO",
            "IGA"
        ],
        "xticks":[
            'Mk01', 'Mk02', 'Mk03', 'Mk04',
            'Mk05', 'Mk06', 'Mk07', 'Mk08', 
            'Mk09', 'Mk10'
        ],
        "line_style" : None,
    }
    assert len(file_dict['algorithms']) == len(file_dict['labels']) 
    assert len(file_dict['paths']) == len(file_dict['xticks']) 
    
    print(f'====== Benchmark dataset1 ===== ')
    print_gap(file_dict, is_avg_gap=True)
    
    # ============ benckmark dataset1 ==============
    file_dict = {
        "paths": [
            result_path+'20230510_212059_test_mt10c1.fjs/log.txt',
            result_path+'20230510_212059_test_mt10cc.fjs/log.txt',
            result_path+'20230510_212059_test_mt10x.fjs/log.txt',
            result_path+'20230510_212059_test_mt10xx.fjs/log.txt',
            result_path+'20230510_212059_test_mt10xxx.fjs/log.txt',
            result_path+'20230510_212059_test_mt10xy.fjs/log.txt',
            result_path+'20230510_212059_test_mt10xyz.fjs/log.txt',
            result_path+'20230510_212059_test_setb4c9.fjs/log.txt',
            result_path+'20230510_212059_test_setb4cc.fjs/log.txt',
            result_path+'20230510_212059_test_setb4x.fjs/log.txt',
            result_path+'20230510_212059_test_setb4xx.fjs/log.txt',
            result_path+'20230510_212059_test_setb4xxx.fjs/log.txt',
            result_path+'20230510_212059_test_setb4xy.fjs/log.txt',
            result_path+'20230510_212059_test_setb4xyz.fjs/log.txt',
            result_path+'20230510_212059_test_seti5cc.fjs/log.txt',
            result_path+'20230510_212059_test_seti5x.fjs/log.txt',
            result_path+'20230510_212059_test_seti5xx.fjs/log.txt',
            result_path+'20230510_212059_test_seti5xxx.fjs/log.txt',
            result_path+'20230510_212059_test_seti5xy.fjs/log.txt',
            result_path+'20230510_212059_test_seti5xyz.fjs/log.txt',
            
            
        ],
        "algorithms":[
            "gtrans_jobcentric_10_6_6_EncV2_DecV5",
            # "dtrans_jobcentric_10_6_6_EncV4_DecV5",
            # "dtrans_jobcentric_10_6_6_EncV6_DecV5",
            # "dtrans_jobcentric_10_6_6_EncV5_DecV5",
            # "dtrans_jobcentric_10_6_6_EncV4_DecV5_sec",
            "matnet_jobcentric_10_6_6",
            "hgnn_jobcentric_10_6_6",
            "dispatch_spt",
            "dispatch_lpt",
            "dispatch_fifo",
            "GA"
        ],
        "labels":[
            "HGS",
            # "HGS (EncV4)",
            # "HGS (EncV6)",
            # "HGS (EncV5)",
            # "HGS (EncV4_2)",
            "MatNet",
            "HGNN",
            "SPT",
            "LPT",
            "FIFO",
            "IGA"
        ],
        "xticks":[
            'mt10c1', 'mt10cc', 'mt10x', 'mt10xx', 'mt10xxx', 'mt10xy', 'mt10xyz',
            'setb4c9', 'setb4cc', 'setb4x', 'setb4xx', 'setb4xxx', 'setb4xy', 'setb4xyz',
            'seti5cc', 'seti5x', 'seti5xx', 'seti5xxx', 'seti5xy', 'seti5xyz'
        ],
        "line_style" : None,
    }
    assert len(file_dict['algorithms']) == len(file_dict['labels']) 
    assert len(file_dict['paths']) == len(file_dict['xticks']) 
    
    print(f'====== Benchmark datset2 ===== ')
    print_gap(file_dict, is_avg_gap=True)
    
    # ===========================================================
    # ======= proposed method analysis : NoEnc, NoDec, self-attention ========
    file_dict = {
        "paths": [
            # result_path+'20230515_092456_test_RL_10_6_6/log.txt',
            
            # result_path+'20230510_211922_test_20_10_10/log.txt',
            # result_path+'20230510_211922_test_30_15_15/log.txt',
            # result_path+'20230510_211922_test_40_20_20/log.txt',
            # result_path+'20230510_211922_test_50_25_25/log.txt',
            result_path+'20230515_131702_test_10_6_6/test_results.pt',
            result_path+'20230515_131702_test_20_10_10/test_results.pt',
            result_path+'20230515_131702_test_30_15_15/test_results.pt',
            result_path+'20230515_131702_test_40_20_20/test_results.pt',
            result_path+'20230515_135054_test_50_25_25/test_results.pt',
        ],
        "algorithms":[
            "gtrans_jobcentric_10_6_6_EncV2_DecV5",
            "dtrans_jobcentric_10_6_6_EncV1_DecV5", # self-attention, ignoring edge attributes
            "dtrans_jobcentric_5_3_3_EncV0_DecV0",
        ],
        "labels":[
            "HGS",  # proposed method should be first index in labels
            "HGS\n(with self-atten)",
            "HGS\n(non-graph)"
        ],
        "xticks":[
            '10X6X6', 
            '20X10X10', 
            '30X15X15', 
            '40X20X20', 
            '50X25X25'
        ],
        "line_style" : None,
    }
    assert len(file_dict['algorithms']) == len(file_dict['labels']) 
    assert len(file_dict['paths']) == len(file_dict['xticks']) 
    
    print(f'====== Proposed method analysis ===== ')
    figure_relative_gap(
        file_dict,
        save_path=save_path
    )
    # ===================================================
    
    file_dict = {
        "paths": [
            [
                result_path+'20230216_080001_test_job10_ma5_veh2/log.txt',
                result_path+'20230216_075524_test_job10_ma5_veh5/log.txt',
                result_path+'20230216_080457_test_job10_ma5_veh10/log.txt',
                result_path+'20230216_080152_test_job10_ma5_veh15/log.txt',
                result_path+'20230216_111314_test_job10_ma5_veh20/log.txt'
            ],
            [
                result_path+'20230216_080934_test_job20_ma10_veh2/log.txt',
                result_path+'20230216_081517_test_job20_ma10_veh5/log.txt',
                result_path+'20230216_081948_test_job20_ma10_veh10/log.txt',
                result_path+'20230216_083903_test_job20_ma10_veh15/log.txt',
                result_path+'20230216_111445_test_job20_ma10_veh20/log.txt',
            ]
        ],
        "algorithms":[
            "HJS_JM_EncV5_DecV4_10_5_5",
            "HJS_JM_EncV5_DecV4_10_5_5",
        ],
        "labels":[
            "Job10_Mch5",
            "Job20_Mch10",
        ],
        "xticks":[
            2, 5, 10, 15, 20
        ],
        # "xticklabels":[
        #     "2",
        #     "15",
        #     "20",
        #     "25"
        # ],
        "line_style" : ['o-'],
    }
    
    # show_result_wrt_graph_size(
    #     file_dict, result='Score:',
    #     xlabel='Number of AGVs', ylabel='Makespan',
    #     graph_style='normal',
    #     same_algo_multi_lines=True,
    #     save_path=os.path.join(save_path, 'makespan_wrt_NumberOfAGVs')
    # )
    
    # show_result_wrt_graph_size(file_dict, result='Score:',
    #     xlabel='Number of AGVs', ylabel='Makespan',
    #     graph_style='normal',
    #     figures=figures
    # )
    