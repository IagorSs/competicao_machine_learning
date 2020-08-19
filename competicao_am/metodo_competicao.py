import pandas as pd
from typing import List

from base_am.metodo import MetodoAprendizadoDeMaquina
from base_am.preprocessamento_atributos import BagOfItems

from .resultado_competicao import ResultadoCompeticao
from .preprocessamento_atributos_competicao import words_IDF, standart_text, gerar_atributos_diretor, zerolistmaker

class MetodoCompeticao(MetodoAprendizadoDeMaquina):

    # Retorna Dataframe 'Comedy' ou 'Action'
    def genero_df(self,df: pd.DataFrame, genero:str):
        return df[df['genero'] == genero]

    def diretores(self,df:pd.DataFrame) -> List[str]:
        max_times = df['dirigido_por'].value_counts()[0]

        for i in range(max_times,0,-1):
            df_diretores = gerar_atributos_diretor(df, i)
            size = len(df_diretores.columns)
            if i == 2 or size > len(df_diretores)/2:
                return df_diretores.columns.to_list()

    def escritores(self, df:pd.DataFrame, min_occur:int = 2) -> List[str]:
        bag = BagOfItems(min_occur=min_occur)
        action_escritores = bag.cria_bag_of_items(df, ['escrito_por_1','escrito_por_2']).columns.to_list()

        return self.clean_list(action_escritores)

    def clean_list(self,my_list:List[str]) -> List[str]:
        if '' in my_list:
            my_list.remove('')
        if 'id' in my_list:
            my_list.remove('id')
        return my_list

    def eval_diretores(self, df:pd.DataFrame, df_to_predict:pd.DataFrame) -> List[str]:
        df_treino_action = self.genero_df(df,'Action')
        df_treino_comedy = self.genero_df(df,'Comedy')

        action_diretores = self.clean_list(self.diretores(df_treino_action))
        comedy_diretores = self.clean_list(self.diretores(df_treino_comedy))

        arr_predict = []

        for value in df_to_predict['dirigido_por']:
            if value in action_diretores:
                arr_predict.append('Action')
            elif value in comedy_diretores:
                arr_predict.append('Comedy')
            else:
                arr_predict.append('default')

        return arr_predict

    def eval_escritores(self, df:pd.DataFrame, df_to_predict:pd.DataFrame) -> List[str]:
        df_treino_action = self.genero_df(df,'Action')
        df_treino_comedy = self.genero_df(df,'Comedy')

        action_escritores = self.clean_list(self.escritores(df_treino_action))
        comedy_escritores = self.clean_list(self.escritores(df_treino_comedy))

        arr_predict = zerolistmaker(len(df_to_predict))

        for i,value in enumerate(df_to_predict['escrito_por_1']):
            if value in action_escritores:
                arr_predict[i] += 1
            elif value in comedy_escritores:
                arr_predict[i] -= 1

        for i, value in enumerate(df_to_predict['escrito_por_2']):
            if value in action_escritores:
                arr_predict[i] += 1
            elif value in comedy_escritores:
                arr_predict[i] -= 1
        
        for i,value in enumerate(arr_predict):
            if value > 0:
                arr_predict[i] = 'Action'
            elif value < 0:
                arr_predict[i] = 'Comedy'
            elif value == 0:
                arr_predict[i] = 'default'
            else:
                raise NameError(f'Break: unexpected value {value} on position {i}')

        return arr_predict

    # Implementar se possível recebimento somente de itens para checar em caso de diretor e escritor serem inconclusivos
    def eval_resumos(self, df:pd.DataFrame, df_to_predict:pd.DataFrame, max_min_IDF:List[float] = False, flags:bool = False) -> List[str]:
        df_treino_action = self.genero_df(df,'Action')
        df_treino_comedy = self.genero_df(df,'Comedy')

        df_to_predict_formated = standart_text(df_to_predict,'resumo')

        data_action_words = words_IDF(df_treino_action,'resumo',flags=flags)

        if flags:
            print('Action words discovered!')

        data_comedy_words = words_IDF(df_treino_comedy,'resumo',flags=flags)

        if flags:
            print('Comedy words discovered!')

        for value_action in data_action_words.keys():
            for value_comedy in data_comedy_words.keys():
                if value_comedy == value_action:
                    data_action_words.drop(value_action)
                    data_comedy_words.drop(value_comedy)
                    break
        
        if flags:
            print('Palavras semelhantes retiradas')

        if not bool(max_min_IDF):
            max_IDF_action = data_action_words[0]*0.8
            max_IDF_comedy = data_comedy_words[0]*0.8

            min_IDF_action = data_action_words[len(data_action_words)-1]
            min_IDF_comedy = data_comedy_words[len(data_comedy_words)-1]
        else:
            max_IDF = max_min_IDF[0]
            min_IDF = max_min_IDF[1]

        if flags:
            print('Max and Min setted')

        action_words = []
        comedy_words = []

        for key,value in data_action_words.iteritems():
            if not bool(max_min_IDF):
                if value <= max_IDF_action and value >= min_IDF_action:
                    action_words.append(key)
            else:
                if value <= max_IDF and value >= min_IDF:
                    action_words.append(key)

        for key,value in data_comedy_words.iteritems():
            if not bool(max_min_IDF):
                if value <= max_IDF_comedy and value >= min_IDF_comedy:
                    comedy_words.append(key)
            else:
                if value <= max_IDF and value >= min_IDF:
                    comedy_words.append(key)

        if flags:
            print('Palavras de comédia e ação refinadas')

        arr_predict = zerolistmaker(len(df_to_predict))

        for i,text in enumerate(df_to_predict_formated['resumo']):
            if type(text) != str or text == '':
                continue
            words = text.split()
            for word in words:
                if word in action_words:
                    arr_predict[i] += 1
                elif word in comedy_words:
                    arr_predict[i] -= 1
                else:
                    continue

        for i,value in enumerate(arr_predict):
            if value > 0:
                arr_predict[i] = 'Action'
            elif value < 0:
                arr_predict[i] = 'Comedy'
            elif value == 0:
                arr_predict[i] = 'default'
            else:
                raise NameError(f'Break: unexpected value {value} on position {i}')

        return arr_predict

    def combine_predictions(self, arrays:List[List[str]]) -> List[str]:

        decided_by_vote = zerolistmaker(len(arrays[0]))

        for array in arrays:

            if len(array) != len(decided_by_vote):
                raise NameError('Break: size of predictions')

            for i,prediction in enumerate(array):
                if prediction == 'default':
                    continue
                elif prediction == 'Action':
                    decided_by_vote[i] += 1
                elif prediction == 'Comedy':
                    decided_by_vote[i] -= 1
                else:
                    raise NameError('Prediction unexpected!')

        for i,value in enumerate(decided_by_vote):
            if value > 0:
                decided_by_vote[i] = 'Action'
            else:
                decided_by_vote[i] = 'Comedy'

        return decided_by_vote

    def eval(self,df_treino:pd.DataFrame, df_data_to_predict:pd.DataFrame, col_classe:str, flags:bool = False) -> ResultadoCompeticao:
        arr_predictions_diretores = self.eval_diretores(df_treino, df_data_to_predict)
        if flags:
            print('Diretores avaliados!')

        arr_predictions_escritores = self.eval_escritores(df_treino, df_data_to_predict)
        if flags:
            print('Escritores avaliados!')

        arr_predictions_resumos = self.eval_resumos(df_treino, df_data_to_predict,flags=flags)
        if flags:
            print('Resumos avaliados!')

        y_to_predict = df_data_to_predict[col_classe].tolist()

        #combina as três
        arr_final_predictions = self.combine_predictions([arr_predictions_diretores,arr_predictions_escritores,arr_predictions_resumos])

        return ResultadoCompeticao(y_to_predict, arr_final_predictions)