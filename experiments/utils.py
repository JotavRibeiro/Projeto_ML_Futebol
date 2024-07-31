import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import os
import mlflow

def plot_metrics(df, 
                 clf, 
                 run,
                 experiment,
                 num_bins=5, 
                 features=None, 
                 prob_col='Probabilidade', 
                 res_col='Res', 
                 xlabel='Faixas de Probabilidade', 
                 ylabel_left='Nº de Partidas', 
                 ylabel_right='% de Ocorrências',
                 bar_alpha=0.6, 
                 figsize=(30, 15), 
                 save_path='//figs//', 
                 file_prefix='metrics', 
                 mlflow_path="classifier"):

    # Criação das faixas de probabilidade
    bins = np.linspace(0, 1, num_bins + 1)
    df[prob_col] = clf.predict_proba(df.loc[:, features])[:, 1]
    labels = [f'[{i/num_bins},{(i+1)/num_bins})' for i in range(num_bins)]
    df['Faixas'] = pd.cut(df[prob_col], bins=bins, labels=labels, include_lowest=True)
    
    # Criação das colunas para cada objetivo
    df['Wins'] = np.where(df[res_col] == 'H', 1, 0)

    # Cálculo das médias e contagens por faixa
    mean_target_by_bin_wins = df.groupby('Faixas')['Wins'].mean().reset_index().dropna()

    count_matches_by_bin = df.groupby('Faixas')[res_col].count().reset_index().dropna()

    # Configuração do estilo do gráfico
    sns.set_style('white')
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot das barras
    bars = sns.barplot(x=count_matches_by_bin['Faixas'], 
                            y=count_matches_by_bin[res_col], 
                            ax=ax1, color='blue', 
                            alpha=bar_alpha, label='')
    
    ax1.set_xlabel(xlabel, fontsize=22, fontweight='bold', labelpad=15)
    ax1.set_ylabel(ylabel_left, color='black', fontsize=22, fontweight='bold', labelpad=15)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=20)
    
    for bar in ax1.patches:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.0f}', 
                ha='center', va='bottom', color='black', fontsize=18, fontweight='bold')

    # Plot das linhas
    ax2 = ax1.twinx()
    line_wins = sns.lineplot(x=mean_target_by_bin_wins['Faixas'], 
                             y=mean_target_by_bin_wins['Wins'], 
                             ax=ax2, color='green', 
                             marker='v', 
                             markersize=16, 
                             markeredgewidth=0, 
                             linewidth=2, 
                             linestyle='--', label='Vitórias')


    ax2.set_ylabel(ylabel_right, color='black', fontsize=22, fontweight='bold', labelpad=30, rotation=270)
    ax2.set_ylim(0, 1)

    percent_formatter = mticker.FuncFormatter(lambda x, pos: '{:.0f}%'.format(x * 100))
    ax2.yaxis.set_major_formatter(percent_formatter)
    ax2.tick_params(axis='y', labelcolor='black', labelsize=20)

    # Plotando números acima das linhas
    for x, y in zip(mean_target_by_bin_wins['Faixas'], mean_target_by_bin_wins['Wins']):
        ax2.text(x, y + 0.01, f'{y:.2%}', ha='center', va='bottom', color='black', fontsize=18, fontweight='bold')

    ax1.tick_params(axis='x', labelcolor='black', labelsize=20)

    legend_elements = [
        Patch(facecolor='blue', edgecolor='blue', label='Nº de Partidas', alpha=bar_alpha),
        Line2D([0], [0], color='green', marker='v', linestyle='--', markersize=12, label='Vitórias'),
    ]

    plt.legend(handles=legend_elements, loc='upper right', fontsize=20, frameon=False, ncol=2)

    plt.title('% DE TRIUNFOS DOS TIMES MANDANTES POR FAIXAS DE PROBABILIDADE DE VITÓRIA', fontsize=24, fontweight='bold', pad=25)

    ax1.grid(axis='y', linestyle=':')

    plot_file = f'{save_path}{file_prefix}_{experiment}_{run}.png'
    plt.savefig(os.getcwd() + plot_file)

    mlflow.log_artifact(os.getcwd() + plot_file, artifact_path=mlflow_path)

    plt.close()
