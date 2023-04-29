import pandas as pd

class Column():
    def __init__(self, name, col: pd.Series) -> None:
        self.col = col
        self.name = name
        self.scale = self.__get_max_decimal_point()
        self.p_c = self.__get_p_c()
    def __get_max_decimal_point(self):
        #calculate max 
        pass
    def __get_p_c(self):
        return (10^self.scale * (self.col.max()-self.col.min())) / len(self.col)
    
    def get_similarity_vector(self):
        pass
        
    def relative_distance(self, v_1, v_2):
        k = min(v_1, v_2) #adopt here how to get minimum of scale
        v_1 = round(v_1, k)
        v_2 = round(v_2, k)
        return (abs(v_1 - v_2))/(round(self.col.max(), k)- round(self.col.min(), k))
    
    def column_similarity(self, v_1, v_2):
        return 1 - self.relative_distance(v_1, v_2)