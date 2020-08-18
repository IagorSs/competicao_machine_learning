import pandas as pd
import math
from base_am.preprocessamento_atributos import BagOfWords, BagOfItems

def standart_text(df:pd.DataFrame, column:str) -> pd.DataFrame:
    df_current = df
    for i,text in enumerate(df_current[column]):
        text_2 = text.replace('.',' ')
        text_final = text_2.replace(',',' ')
        df_current[column][i] = str.upper(text_final)
        if i == len(df):
            break
    return df_current

def words_IDF(dataFrame:pd.DataFrame, column:str, min_ocurr:int = 4) -> pd.Series:
    data_words = pd.Series()

    df = standart_text(dataFrame, column)

    for i,text in enumerate(df[column]):
        words = text.split()
        for word in words:
            if len(word) < min_ocurr:
                continue
            no_check = False
            for key in data_words.index:
                if word == key:
                    no_check = True
                    break
            if no_check:
                continue
            data_words[word] = calcula_IDF(df,word,column)
        if i == len(df):
            break
            
    return data_words

def calcula_IDF(df:pd.DataFrame, palavra:str, column:str) -> float:
    N = len(df)
    ni = 0
    
    for i,text in enumerate(df[column]):
        words = text.split()
        for word in words:
            if palavra == word:
                ni += 1
                break
        if i == len(df):
            break

    if ni == 0:
        return 0
    return math.log(N/ni)

def gerar_atributos_ator(df_treino:pd.DataFrame, df_data_to_predict: pd.DataFrame) -> pd.DataFrame:
    obj_bag_of_actors = BagOfItems(min_occur=3)
    df_treino_boa = obj_bag_of_actors.cria_bag_of_items(df_treino,["ator_1","ator_2","ator_3","ator_4","ator_5"])
    df_data_to_predict_boa = obj_bag_of_actors.aplica_bag_of_items(df_data_to_predict,["ator_1","ator_2","ator_3","ator_4","ator_5"])

    

    return df_treino_boa, df_data_to_predict_boa

def gerar_atributos_resumo(df_treino:pd.DataFrame, df_data_to_predict: pd.DataFrame) -> pd.DataFrame:
    bow_amostra = BagOfWords()
    df_bow_treino = bow_amostra.cria_bow(df_treino,"resumo")
    df_bow_data_to_predict = bow_amostra.aplica_bow(df_data_to_predict,"resumo")

    return df_bow_treino,df_bow_data_to_predict
