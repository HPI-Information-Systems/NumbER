import pandas as pd
import numpy as np    

class Column():
    def __init__(self, name, col: pd.Series) -> None:
        col = col.astype(float)
        self.col = col
        self.name = name
        self.scale = self.__get_max_decimal_point()
        self.p_c = self.__get_p_c()
        
    def get_amount_of_decimal_points(self, value: float):
        length = str(value).split(".")[-1]
        return len(length) if length != "nan" else 0
    def __get_max_decimal_point(self):
        #calculate max 
        amount_of_decimal_points = map(self.get_amount_of_decimal_points, self.col)
        return float(max(amount_of_decimal_points))
    
    def __get_p_c(self):
        if type(self.col.max()) != np.float64 and type(self.col.max()) != float:
            return 0
        print("s",self.scale)
        return (pow(10,self.scale) * (self.col.max()-self.col.min())) / len(self.col)
        #return s
    
    def get_similarity_vector(self):
        pass
        
    def relative_distance(self, v_1, v_2):
        k = min(self.get_amount_of_decimal_points(v_1), self.get_amount_of_decimal_points(v_2)) #adopt here how to get minimum of scale
        v_1 = round(v_1, k)
        v_2 = round(v_2, k)
        return (abs(v_1 - v_2))/(round(self.col.max(), k)- round(self.col.min(), k))
    
    def column_similarity(self, v_1, v_2):
        return 1 - self.relative_distance(v_1, v_2)