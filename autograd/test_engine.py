from engine import Value 

def test_ops():
    a=Value(2.2)
    b=Value(3.5)
    c=Value(8.5)
    d=Value(-4.5)
    #normal add 
    e=a+b;e.label='e'
    print(e,'-->',e.label,'children -->',e._prev,'-->',e._op)
    #add Value+constant
    f=c+2;f.label='f'
    print(f,'-->',f.label,'children -->',f._prev,'-->',f._op)
    #radd
    g=5+f;g.label='g'
    print(g,'-->',g.label,'children -->',g._prev,'-->',g._op)
    #sub,negation,mul
    h=d-c;h.label='h'
    print(h,'-->',h.label,'children -->',h._prev,'-->',h._op)
    # normal mul
    k=f*g;k.label='k'
    print(k,'-->',k.label,'children -->',k._prev,'-->',k._op)
    #rmul
    l=2*f;l.label='l'
    print(l,'-->',l.label,'children -->',l._prev,'-->',l._op)

test_ops()

     
