from base_am.avaliacao import OtimizacaoObjetivo
from base_am.metodo import MetodoAprendizadoDeMaquina
from base_am.resultado import Fold, Resultado
from competicao_am.metodo_competicao import MetodoCompeticaoProf
import optuna
from competicao_am.preprocessamento_atributos_competicao import gerar_atributos_diretor
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


class OtimizacaoObjetivoSVMCompeticao(OtimizacaoObjetivo):
    def __init__(self, fold:Fold, num_arvores_max:int=5):
        super().__init__(fold)
        self.num_arvores_max = num_arvores_max

    def obtem_metodo(self,trial: optuna.Trial)->MetodoAprendizadoDeMaquina:
        #Um custo adequado para custo pode variar muito, por ex, para uma tarefa 
        #o valor de custo pode ser 10, para outra, 32000. 
        #Assim, normalmente, para conseguir valores mais distintos,
        #usamos c=2^exp_cost
        exp_cost = trial.suggest_uniform('min_samples_split', 0, 7) 

        scikit_method = LinearSVC(C=2**exp_cost, random_state=2)

        return MetodoCompeticaoProf(scikit_method)

    def resultado_metrica_otimizacao(self,resultado: Resultado) -> float:
        return resultado.macro_f1


class MeuMetodo(MetodoAprendizadoDeMaquina):

    def eval(self,df_treino:pd.DataFrame, df_data_to_predict:pd.DataFrame, col_classe:str) -> Resultado:
        return None

    def data_rept(self,data_frame_treino:pd.DataFrame) -> pd.DataFrame:
        max_times = data_frame_treino['dirigido_por'].value_counts()[0]

        for i in range(max_times,0,-1):
            size = len(gerar_atributos_diretor(data_frame_treino, i).columns)
            if i == 2 or size > len(gerar_atributos_diretor(data_frame_treino, i))//2:
                return gerar_atributos_diretor(data_frame_treino, i)
    
    def obtem_y(self, df_treino:pd.DataFrame, df_data_to_predict:pd.DataFrame, col_classe:str):
        y_treino = self.class_to_number(df_treino[col_classe])
        y_to_predict = None
        #y_to_predict pod não existir (no dataset de teste fornecido pelo professor, por ex)
        if col_classe in df_data_to_predict.columns:
            y_to_predict = self.class_to_number(df_data_to_predict[col_classe])
        return y_treino,y_to_predict

    def obtem_x(self, df_treino:pd.DataFrame, df_data_to_predict:pd.DataFrame, col_classe:str):
        x_treino = df_treino.drop(col_classe, axis = 1)
        x_to_predict = df_data_to_predict
        if col_classe in df_data_to_predict.columns:
            x_to_predict = df_data_to_predict.drop(col_classe, axis = 1)
        return x_treino, x_to_predict