from engine import Value 
import math
def test_ops():
    a=Value(2.2)
    b=Value(3.5)
    c=Value(8.5)
    d=Value(-4.5)
    #normal add 
    e=a+b;e.label='e'
    assert e.data==(a.data+b.data),'incorrect value returned'
    print(e,'-->',e.label,'children -->',e._prev,'-->',e._op)
    #add Value+constant
    f=c+2;f.label='f'
    assert f.data==(c.data+2),'incorrect value returned'
    print(f,'-->',f.label,'children -->',f._prev,'-->',f._op)
    #radd
    g=5+f;g.label='g'
    assert g.data==(5+f.data),'incorrect value returned'
    print(g,'-->',g.label,'children -->',g._prev,'-->',g._op)
    #sub,negation,mul
    h=d-c;h.label='h'
    assert h.data==(d.data-c.data),'incorrect value returned'
    print(h,'-->',h.label,'children -->',h._prev,'-->',h._op)
    # normal mul
    k=f*g;k.label='k'
    assert k.data==(f.data*g.data),'incorrect value returned'
    print(k,'-->',k.label,'children -->',k._prev,'-->',k._op)
    #rmul
    l=2*f;l.label='l'
    assert l.data==(2*f.data),'incorrect value returned'
    print(l,'-->',l.label,'children -->',l._prev,'-->',l._op)
    m=l**2;m.label='m'
    assert m.data==(l.data**2),'incorrect value returned'
    print(m,'-->',m.label,'children -->',m._prev,'-->',m._op)
    n=m/2;m.label='m'
    assert n.data==(m.data/2),'incorrect value returned'
    print(n,'-->',n.label,'children -->',n._prev,'-->',n._op)
    q=m/k;q.label='q'
    assert q.data==(m.data/k.data),'incorrect value returned'
    print(q,'-->',q.label,'children -->',q._prev,'-->',q._op)
    o=q.tanh();o.label='o'
    assert o.data==(math.exp(2*q.data)-1)/(math.exp(2*q.data)+1)
    print(o,'-->',o.label,'children -->',o._prev,'-->',o._op)
test_ops()

     
