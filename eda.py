import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def summary_stat(dataframe):
    print(dataframe.info())
    print(dataframe.describe())

def pairplot(dataframe,xvars,yvars,hue,counter):
    #plt.figure(num=counter);
    counter+=1
    fig = sns.pairplot(data=dataframe,hue=hue,x_vars=xvars,y_vars=yvars)
    fig.savefig('pairplot.jpg')
    plt.clf()
    return counter

def plot_corr(dataframe,counter):
    #plt.figure(num=counter);
    counter+=1
    fig = sns.heatmap(dataframe.corr(),annot=True).get_figure()
    fig.savefig('correlation.jpg')
    plt.clf()
    return counter

def plot_violin(dataframe,counter):
    counter+=1
    fig = sns.violinplot(data=dataframe,split=True,inner='quartile').get_figure()
    fig.savefig('violinplot.jpg')
    plt.clf()
    return counter

def plot_box(dataframe,counter):
    counter+=1
    fig = sns.boxplot(data=dataframe).get_figure()
    fig.savefig('boxplot.jpg')
    plt.clf()
    return counter

def explore_data(data):
    summary_stat(data)
    counter=0
    counter = pairplot(dataframe=data,xvars=list(data.columns[:2])+list(data.columns[-3:-1]),yvars=data.columns[2],
                        hue=data.columns[-1],counter=counter)
    counter = plot_corr(dataframe=data,counter=counter)
    counter = plot_violin(dataframe=data,counter=counter)
    counter = plot_box(dataframe=data,counter=counter)

