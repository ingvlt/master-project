import os
from sklearn.metrics import f1_score
import pandas as pd
import seaborn as sns

# originally code used in notebook and not as python file

#data from gold data, female% - male% > 0 => female-dominated, female% - male% < 0 => male-dominated
y_true = pd.read_csv('gender_equality_ssb.csv', ';', header=0)
df = pd.DataFrame(columns=['f1', 'dataset', 'model'])

# make heatmap for bias
for root, dirs, files in os.walk('./bias'):
    for file in files:
        #last inn data frå språkmodell
        # load data from language model
        y_pred = pd.read_csv(os.path.join(root, file), header=0)
        res = f1_score(y_true['binary'], y_pred['bias_score'], average='macro')
        words = file.split('.')
        filename = words[0].split('_')
        data = {'f1': res,
                'dataset': filename[1],
                'model': filename[2],
                }
        df.loc[len(df)] = data
print(df)
sns.set_theme()
d = df.pivot('dataset', 'model', 'f1')
sns.heatmap(d, annot=True, fmt=".2f", cmap='crest')


# make heatmap for accuracy in pos 
data = pd.read_csv('/content/accuracy.csv', header=0)

sns.set_theme()
d = data.pivot('dataset', 'model', 'accuracy')
sns.heatmap(d, annot=True, fmt=".2f", cmap='crest')