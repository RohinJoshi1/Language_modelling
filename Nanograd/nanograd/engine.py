
class Value:
    #Stores value , grad

    def __init__(self,data,_children=(),_op = ''):
        self.data = data
        self.grad = 0
        self._backward = lambda:None
        self._prev = set(_children)
        self._op = _op 

    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data, (self,other),'+')

        def _backward():
            self.grad += out.grad
            other.grad +=out.grad 
        out._backward = _backward

        return out

    def __pow__(self,other):
        assert isinstance(other,(int, float)),"exp must be int or float"
        out = Value(self.data**other, (self,),f'**{other}')

        def _backward():
            self.grad += (other*self.data**(other-1))*other.grad
        out._backward = _backward
        return out


    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, (self,other),'*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad 
        out._backward = _backward

        return out
    
    def relu(self):
        out = Value(0 if self.data <0 else self.data,(self,),'ReLu')

        def _backward():
            self.grad += (out.data >0 )* out.grad 
        out._backward = _backward

        return out 
    #Now we define the backprop 

    def backward(self):

        topsort = [] 
        visited = set()
        def tsort(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    tsort(child)
                topsort.append(v)
        tsort(self)

        self.grad = 1
        for v in reversed(topsort):
            v._backward() 
        
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

