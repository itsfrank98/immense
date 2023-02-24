import pickle
import pandas as pd
with open('textual_data/list_counter.pickle', 'rb') as handle:
    X_test = pickle.load(handle)


list_s = []
list_i = []
i = 1
for x in X_test:
    temp = ' '.join(x)
    list_i.append(i)
    list_s.append(temp)
    i = i + 1

d = {'id': list_i, 'text_cleaned': list_s}
df = pd.DataFrame(data=d)
df.to_csv('counter_final.csv', index=False, sep='\t')
