import pandas as pd


Y1 = pd.read_csv("Y1.csv")
Y1['left_right'] = Y1['lid'].astype(str) + Y1['rid'].astype(str)
Y2 = pd.read_csv("Y2.csv")
Y2['left_right'] = Y2['lid'].astype(str) + Y2['rid'].astype(str)
reference = Y1.append(Y2)['left_right'].values
reference1 = Y1['left_right'].values
reference2 = Y2['left_right'].values



output = pd.read_csv("output.csv")
output['left_right'] = output['left_instance_id'].astype(str) + output['right_instance_id'].astype(str)

output_values = output['left_right'].values
inter1 = set.intersection(set(output_values), set(reference1))
inter2 = set.intersection(set(output_values), set(reference2))
inter = set.intersection(set(output_values), set(reference))

recall1 = len(inter1) / len(reference1)
recall2 = len(inter2) / len(reference2)
recall = len(inter) / len(reference)
print(f'Recall X1: {recall1}.\nRecall X2: {recall2}.\nRecall Overall: {recall}.\n')
