from engine import Value 

def test_ops():
    a=Value(2.2)
    b=Value(3.5)
    c=Value(8.5)
    d=Value(-4.5)
    #normal add 
    e=a+b;e.label='e'
    print(e)
    #add Value+constant
    f=c+2;f.label='f'
    print(f)
    #reverse add
    g=5+f;g.label='g'
    print(g)
    h=d-c;h.label='h'
    print(h)
    k=f*g;k.label='k'

test_ops()

     
