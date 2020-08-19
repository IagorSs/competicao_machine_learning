from base_am.resultado import Resultado
from typing import List
from .preprocessamento_atributos_competicao import zerolistmaker

class ResultadoCompeticao(Resultado):

    @staticmethod
    def convert_list_to_int(data_input:List[str]) -> List[int]:
        output = zerolistmaker(len(data_input))

        for i,value in enumerate(data_input):
            if value == 'Comedy':
                output[i] = 1
            elif value == 'Action':
                output[i] = 2
            else:
                raise NameError(f'Unexpectted value for Resultado - {value}')

        return output

    def __init__(self, y:List[str], predict_y:List[str]):
        super().__init__(y,predict_y)
        self.y = self.convert_list_to_int(y)
        self.predict_y = self.convert_list_to_int(predict_y)