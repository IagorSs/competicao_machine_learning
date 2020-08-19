from competicao_am.metodo_competicao import MetodoCompeticao
import pandas as pd

def gerar_saida_teste( df_data_to_predict:pd.DataFrame, col_classe:str, num_grupo:str):
    """
    Assim como os demais códigos da pasta "competicao_am", esta função 
    só poderá ser modificada na fase de geração da solução. 
    """

    #o treino será sempre o dataset completo - sem nenhum dado a mais e sem nenhum preprocessamento
    #esta função que deve encarregar de fazer o preprocessamento

    if df_data_to_predict == None:
        raise NameError('Não foi passado nenhum dataFrame para ser predito')

    if num_grupo == None or type(num_grupo) != str:
        raise NameError(f'Num_grupo inadequado (lembre-se que ele deve ser uma string) - {num_grupo}')

    df_treino = pd.read_csv("datasets/movies_amostra.csv")

    ml_method = MetodoCompeticao()

    arr_predictions_diretores = ml_method.eval_diretores(df_treino, df_data_to_predict)
    arr_predictions_escritores = ml_method.eval_escritores(df_treino, df_data_to_predict)
    arr_predictions_resumos = ml_method.eval_resumos(df_treino, df_data_to_predict)

    #combina as três
    arr_final_predictions = ml_method.combine_predictions([arr_predictions_diretores,arr_predictions_escritores,arr_predictions_resumos])

    #grava o resultado obtido
    with open(f"predict_grupo_{num_grupo}.txt","w") as file_predict:
        for predict in arr_final_predictions:
            file_predict.write(f'{predict}\n')
