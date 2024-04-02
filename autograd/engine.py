class Value:
    def __init__(self,data,_children=()):
        self.data=data
        self._prev=set(_children)
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __str__(self):
        return f"{self.data}"
    