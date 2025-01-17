import pandas as pd

def get_data(file_path):
    food_df = pd.read_excel(file_path, sheet_name='식품유형')
    danger_df = pd.read_excel(file_path, sheet_name='원인요소')

    return food_df, danger_df

def get_label_vocab(df, label_name):
    with open(f'{label_name}_label.vocab', 'w', encoding='euc-kr') as f:
        f.write('Root\n')
        for _, row in df.iterrows():
            line = row['대분류'] + '/' + row['중분류'] + '/' + row['소분류']
            f.write(line+'\n')

def vocab_to_taxonomy(vocab_path, label_name):
    with open(vocab_path, 'r', encoding='euc-kr') as f:
        vocab_lines = f.readlines()

    hierarchy_dict = {}
    for line in vocab_lines:
        line = line.strip()
        if line:
            parts = line.split('/')
            for i in range(len(parts)):
                parent = '/'.join(parts[:i])
                child = '/'.join(parts[:i+1])
                if parent not in hierarchy_dict:
                    hierarchy_dict[parent] = []
                if child not in hierarchy_dict[parent]:
                    hierarchy_dict[parent].append(child)

    taxonomy_path = f'{label_name}_taxo.taxonomy'
    with open(taxonomy_path, 'w', encoding='euc-kr') as f:
        for parent, children in hierarchy_dict.items():
            if parent:
                f.write(f'{parent}\t{'\t'.join(children)}\n')

def __main__():
    file_path = '식품유형 및 원인요소 분류표.xlsx'
    food_df, danger_df = get_data(file_path)
    get_label_vocab(food_df, 'food')
    vocab_to_taxonomy('food_label.vocab', 'food')
    get_label_vocab(danger_df, 'danger')
    vocab_to_taxonomy('danger_label.vocab', 'danger')

if __name__ == '__main__':
    __main__()