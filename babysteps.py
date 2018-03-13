import numpy
import theano.tensor as T
from theano import function
import theano


#variables are symbols now
x = T.dscalar('x')
#0 dim arrays of doubles
y = T.dscalar('y')

z = x + y

#can use eval
f = function([x,y], z)

#output is a numpyt.darray with 0 dim
f(2,3)

# Subsequent calls to eval() on that same variable will be fast, because the variable caches the compiled function.
print z.eval({x : 16.3, y : 12.1})

# print numpy.allclose(z.eval({x : 16.3, y : 12.1}), 28.4)

x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)

print f([[1, 2], [3, 4]], [[10, 20], [30, 40]])

#It is possible to add scalars to matrices, vectors to matrices, scalars to vectors, etc. The behavior of these operations is defined by broadcasting.

a = theano.tensor.vector() # declare variable
b = theano.tensor.vector() # declare variable
out = a ** 2 + b ** 2 + 2 * a * b               # build symbolic expression
f = theano.function([a, b], out)   # compile function
print(f([ 1, 2],[3,4]))

#element wise operations

a, b = T.dmatrices('a', 'b')

diff = a - b

abs_diff = abs(diff)

f = function([a,b],diff)

print(f([[1,13]],[[11,1]]))