import re
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def summarise_results(path):
    with open(path, 'r') as file:
            text = file.read()
    print(text)
    loss_regex = r"INFO:root:Test After Training =.*"
    loss_match = re.findall(loss_regex,text)
    if loss_match == []:
        loss_regex = r"INFO:root:Test After Training:	 Average:.*"
        loss_match = re.findall(loss_regex,text)
#     print(loss_match)
    metric_regex=r'\w+@\w+'
    metric_match = re.findall(metric_regex,loss_match[0])
    
    val_regex=r'\d+.\d+'
    val_match = re.findall(val_regex,loss_match[0])
    print(f'Final test results are as follows-')
    for i,j in zip(metric_match,val_match):
        print(f'{i} - {j}')

    print(f'The Best AUC results are as follows-')
    loss_regex = r"INFO:root:u_occupation best AUC.*"
    loss_match = re.findall(loss_regex,text)
    loss_regex=r'\d+.\d+'
    loss_match = re.findall(loss_regex,loss_match[0])
    print(f'Occupation AUC - {float(loss_match[0])}')

    loss_regex = r"INFO:root:u_gender best AUC.*"
    loss_match = re.findall(loss_regex,text)
    loss_regex=r'\d+.\d+'
    loss_match = re.findall(loss_regex,loss_match[0])
    print(f'Gender AUC - {float(loss_match[0])}')

    loss_regex = r"INFO:root:u_age best AUC.*"
    loss_match = re.findall(loss_regex,text)
    loss_regex=r'\d+.\d+'
    loss_match = re.findall(loss_regex,loss_match[0])
    print(f'Age AUC - {float(loss_match[0])}')


def plot_loss(path,title):
    with open(path, 'r') as file:
        text = file.read()
    loss_regex = r"INFO:root:loss = \d+.\d+"
    loss_match = re.findall(loss_regex,text)
    loss_fn=r'\d+.\d+'
    loss=[]
    for i in loss_match:
        text=re.findall(loss_fn,i)
        loss.append(float(text[0]))
    fig = plt.figure()
    sns.set(font="Verdana")
    plt.plot(range(1,len(loss)+1),loss)
    fig.suptitle(title, fontsize=12)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.savefig(title+'.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, help="Path to log config")
    parser.add_argument("-title", type=str, help="title for plots")

    args = parser.parse_args()

    plot_loss(args.path,args.title)
    summarise_results(args.path)