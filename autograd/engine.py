class Value:
    def __init__(self,data,_children=(),_op='',label=''):
        self.data=data
        self._prev=set(_children)
        self._op=_op
        self.label=label
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        out=Value(self.data+other.data,(self,other),'+')
        return out 
    
    def __radd__(self,other):
        return self+other
    
    def __mul__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        out=Value(self.data*other.data,(self,other),'*')
        return out
    
    def __rmul__(self,other):
        return self*other
    
    def __neg__(self):
        return self *-1
    
    def __sub__(self,other):
        out=self+(-other)
        return out 
    def __pow__(self,other):
        assert isinstance(other,(int,float)),'Value is not supported as a power, use a float or an int'
        out=Value(self.data**other,(self,),f'**{other}')
        return out 
    def __truediv__(self,other):
        return self * other**-1
