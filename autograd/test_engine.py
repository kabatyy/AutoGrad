from engine import Value 

def test_ops():
    a=Value(2.2)
    b=Value(4.5)
    return a+b
     
c=test_ops()
print(c._prev)