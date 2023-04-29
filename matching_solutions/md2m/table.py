from column import Column
import typing

class Table():
    def __init__(self, name: str, cols: typing.List(Column)) -> None:
        self.name = name
        comparison_column_threshold = 0.05
        def is_above_threshold(el: Column):
            return True if el.p_c > comparison_column_threshold else False
        self.comparison_cols = filter(is_above_threshold, cols)
        pass
    
    def similarity(self, a, b):
        sum = 0
        for column in self.comparison_cols:
            a = a[column.name]
            b = b[column.name]
            sum += column.p_c * column.column_similarity(a,b)
        return sum / len(self.comparison_cols)