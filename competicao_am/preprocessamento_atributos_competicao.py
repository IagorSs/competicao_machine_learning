import pandas as pd
import math
from typing import List
from base_am.preprocessamento_atributos import BagOfWords, BagOfItems

def standart_text(df:pd.DataFrame, column:str) -> pd.DataFrame:
    df_current = df
    index = df.index
    for i,text in enumerate(df_current[column]):
        if type(text) != str or text == '':
            df_current[column][index[i]] = str('')
            continue
        text_2 = text.replace('.',' ')
        text_3 = text_2.replace('"',' ')
        text_final = text_3.replace(',',' ')
        df_current[column][index[i]] = str.upper(text_final)
    return df_current

def words_IDF(dataFrame:pd.DataFrame, column:str, min_lenght:int = 4, flags:bool = False) -> pd.Series:
    data_words = pd.Series()

    df = standart_text(dataFrame, column)

    for i,text in enumerate(df[column]):
        if i % 100 == 0 and flags:
            print(f'{i}/{len(df[column])}')
        if type(text) != str or text == '':
            continue
        words = text.split()
        for word in words:
            if len(word) < min_lenght:
                continue
            no_check = False
            for key in data_words.index:
                if word == key:
                    no_check = True
                    break
            if no_check:
                continue
            data_words[word] = calcula_IDF(df,word,column)
            
    return data_words

def calcula_IDF(df:pd.DataFrame, palavra:str, column:str) -> float:
    N = len(df)
    ni = 0
    
    for text in df[column]:
        if type(text) != str:
            continue
        words = text.split()
        for word in words:
            if palavra == word:
                ni += 1
                break

    if ni == 0:
        return 0
    return math.log(N/ni)

def gerar_atributos_diretor(df_treino:pd.DataFrame, min_occur:int = 0) -> pd.DataFrame:
    obj_bag = BagOfItems(min_occur=min_occur)
    df_treino_boa = obj_bag.cria_bag_of_items(df_treino,["dirigido_por"])
    
    return df_treino_boa

def zerolistmaker(n:int) -> List[int]:
    listofzeros = [0] * n
    return listofzeros