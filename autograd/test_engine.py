from engine import Value 

def test_ops():
    #normal add 
    a=Value(2.2)
    b=Value(4.5)
    c=a+b
    print(c)
    #add Value+constant
    d=Value(2.0)+5
    print(d)
    #reverse add
    e=5+Value(3.0)
    print(e)

test_ops()

     
