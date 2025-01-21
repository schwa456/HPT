import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import json

"""
Food Safety Data are from ROK MFDS(Ministry of Food and Drug Safety)
"""

def get_data(file_path, year):
    df = pd.read_excel(file_path, sheet_name=year)
    return df

def separate_label(df, label_name='원인요소'):
    label = df[label_name]
    split_columns = label.str.split('>', expand=True)
    split_columns.columns = [f'{label_name}_{i+1}' for i in range(split_columns.shape[1])]
    df = pd.concat([df, split_columns], axis=1)

    return df

def read_db(df):
    X = df[['제목', '내용']]
    food_y = df[['식품등유형']]
    danger_y = df[['원인요소']]
    return X, food_y, danger_y

def get_trains_tests():
    years = [
        '2014'#, '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'
        ]
    X_trains, X_tests, y_trains, y_tests = [], [], [], []
    for year in years:
        df = pd.read_excel('식품안전정보DB-url 추가(2014~2023).xls', sheet_name=year)
        X, food_y, danger_y = read_db(df)

        X_train, X_test, y_train, y_test = train_test_split(X, danger_y, test_size=.2, random_state=42)
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)

    return X_trains, X_tests, y_trains, y_tests

def __main__():
    label_dict = {}
    hiera = defaultdict(set)
    year = '2014'
    df = pd.read_excel('식품안전정보DB-url 추가(2014~2023).xls', sheet_name=year)
    df = separate_label(df)
    doc_labels = []
    for _, row in df.iterrows():
        doc_label = list(row[['원인요소_1', '원인요소_2', '원인요소_3']])
        doc_labels.append(doc_label)
    for l in doc_labels:
        if l[0] not in label_dict:
            label_dict[l[0]] = len(label_dict)
    for l in doc_labels:
        assert len(l) == 3
        if l[1] not in label_dict:
            label_dict[l[1]] = len(label_dict)
        hiera[label_dict[l[0]]].add(label_dict[l[1]])
    for l in doc_labels:
        assert len(l) == 3
        if l[2] not in label_dict:
            label_dict[l[2]] = len(label_dict)
        hiera[label_dict[l[1]]].add(label_dict[l[2]])
    value_dict = {i: v for v, i in label_dict.items()}
    torch.save(value_dict, 'value_dict.pt')
    torch.save(hiera, 'slot.pt')

    id = [i for i in range(len(df))]
    np_data = np.array(id)
    np.random.shuffle(id)
    np_data = np_data[id]
    train, test = train_test_split(np_data, test_size=0.2, random_state=0)
    train, val = train_test_split(train, test_size=0.2, random_state=0)
    train = train.tolist()
    val = val.tolist()
    test = test.tolist()

    with open('FoodSafety_train.json', 'w') as f:
        for i in train:
            line = json.dumps({'token': df.iloc[i].to_dict(), 'label': [label_dict[j] for j in doc_labels[i]]})
            f.write(line + '\n')
    with open('FoodSafety_dev.json', 'w') as f:
        for i in val:
            line = json.dumps({'token': df.iloc[i].to_dict(), 'label': [label_dict[j] for j in doc_labels[i]]})
            f.write(line + '\n')
    with open('FoodSafety_test.json', 'w') as f:
        for i in test:
            line = json.dumps({'token': df.iloc[i].to_dict(), 'label': [label_dict[j] for j in doc_labels[i]]})
            f.write(line + '\n')



if __name__ == '__main__':
    __main__()