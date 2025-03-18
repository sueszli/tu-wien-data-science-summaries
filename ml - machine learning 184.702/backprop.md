- https://www.youtube.com/watch?v=AXqhWeUEtQU
- https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- https://www.3blue1brown.com/lessons/backpropagation-calculus
- https://www.3blue1brown.com/lessons/backpropagation

# partial derivative

*derivative*

- $\begin{aligned} L=\lim_{h\to0}\frac{f(a+h)-f(a)}h \end{aligned}$
- chain rule:
	- assuming $F(x)=f(g(x))$
	- $F^\prime(x)=f^\prime(g(x)) \cdot g^\prime(x)$

*partial derivative*

- $\Large \frac{\partial f(x)}{\partial x}$ = how does a tiny nudge in $x$ change $f(x)$?
- chain rule:
	- $y = f(g(t))$
	- $x = g(t)$
	- $\Large \frac{dy}{dt}=\frac{dy}{dx}\frac{dx}{dt}$

# simplified backpropagation

![](assets/Pasted%20image%2020240411175937.png)

![|300](assets/Pasted%20image%2020240411175948.png)

assume we have a network where all layers have exactly 1 node with 1 weight and 1 bias.

we want to know how the weights impact the cost for a single training sample.

so we apply the chain rule for partial derivatives:

- $\begin{gathered}\frac{\partial {\color{OrangeRed} C_0}}{\partial w^{(L)}}=\frac{\partial {\color{CarnationPink} z^{(L)}} }{\partial w^{(L)}} \frac{\partial {\color{CornflowerBlue} a^{(L)}} }{\partial {\color{CarnationPink} z^{(L)}} } \frac{\partial {\color{OrangeRed} C_0}}{\partial {\color{CornflowerBlue} a^{(L)}} } =  {\color{CornflowerBlue} a^{(L-1)}} ~\cdot~ \sigma'({\color{CarnationPink} z^{(L)}} ) ~\cdot~ 2({\color{CornflowerBlue} a^{(L)}}-y) \end{gathered}$
- $\begin{gathered}\frac{\partial {\color{OrangeRed} C_0}}{\partial b^{(L)}}=\frac{\partial {\color{CarnationPink} z^{(L)}} }{\partial b^{(L)}} \frac{\partial {\color{CornflowerBlue} a^{(L)}} }{\partial {\color{CarnationPink} z^{(L)}} } \frac{\partial {\color{OrangeRed} C_0}}{\partial {\color{CornflowerBlue} a^{(L)}} } = 1 ~\cdot~ \sigma'({\color{CarnationPink} z^{(L)}} ) ~\cdot~ 2({\color{CornflowerBlue} a^{(L)}}-y) \end{gathered}$
- $\begin{gathered}\frac{\partial {\color{OrangeRed} C_0}}{\partial {\color{CornflowerBlue} a^{(L-1)}}}=\frac{\partial {\color{CarnationPink} z^{(L)}} }{\partial b^{(L)}} \frac{\partial {\color{CornflowerBlue} a^{(L)}} }{\partial {\color{CarnationPink} z^{(L)}} } \frac{\partial {\color{OrangeRed} C_0}}{\partial {\color{CornflowerBlue} a^{(L)}} } = w^{(L)} ~\cdot~ \sigma'({\color{CarnationPink} z^{(L)}} ) ~\cdot~ 2({\color{CornflowerBlue} a^{(L)}}-y) \end{gathered}$

where:

- ${\color{OrangeRed} C_0(\dots)} = ({\color{CornflowerBlue} a^{(L)}} - y)^2$ 
	- $y$ = desired output value in training sample $0$
	- $a^{(L)}$ = actual output at layer $L$, which is the output layer
	- $C_0(\dots)$ =  squared error as cost function for training sample $0$
	- the argument of the cost function are all weights and biases of this neural network
- ${\color{CornflowerBlue} a^{(L)}}= \sigma ({\color{CarnationPink} z^{(L)}}) = \sigma (~w^{(L)} \cdot {\color{CornflowerBlue} a^{(L-1)}} + b^{(L)}~)$
	- $\sigma$ = some activation function like the sigmoid function
	- $w^{(L)}$ = weight of layer $L$
	- $b^{(L)}$ = biase of level

then we then calculate the average cost across all training examples:

- $\begin{gathered}\frac{\partial  C}{\partial w^{(L)}}=\frac1n\sum_{k=0}^{n-1}\frac{\partial {\color{OrangeRed} C_k}}{\partial w^{(L)}}\end{gathered}$

and then repeat this process for all previous layers by traversing the dependency tree, i.e:

- $\begin{gathered}\frac{\partial C_0}{\partial w^{(L-1)}}=\frac{\partial z^{(L-1)}}{\partial w^{(L-1)}}\frac{\partial a^{(L-1)}}{\partial z^{(L-1)}}\frac{\partial z^{(L)}}{\partial a^{(L-1)}}\frac{\partial a^{(L)}}{\partial z^{(L)}}\frac{\partial C_0}{\partial a^{(L)}}\end{gathered}$
- $\dots$

this way we get the complete gradient vector, which we use to update the weights and biases in a gradient descent kind of way:

- $\large \nabla C = \begin{bmatrix} \frac{\partial C}{\partial w^{(1)}} \\ \frac{\partial C}{\partial b^{(1)}} \\ ... \\ \frac{\partial C}{\partial w^{(L)}} \\ \frac{\partial  C}{\partial b^{(L)}} \end{bmatrix}$

# backpropagation

now each layer can have multiple nodes:

- $a_k^{(L-1)} ~ – ~ w^{(L)}_{jk} \longrightarrow a_j^{(L)}$

we apply the chain rule for partial derivatives to get the cost for a single training sample:

- $\begin{gathered}\frac{\partial C_0}{\partial w_{jk}^{(L)}}=\frac{\partial z_j^{(L)}}{\partial w_{jk}^{(L)}}\frac{\partial a_j^{(L)}}{\partial z_j^{(L)}}\frac{\partial C_0}{\partial a_j^{(L)}}\end{gathered}$
- $\begin{gathered}\frac{\partial C_0}{\partial a_k^{(L-1)}}=\sum_{j=0}^{n_L-1}\frac{\partial z_j^{(L)}}{\partial a_k^{(L-1)}}\frac{\partial a_j^{(L)}}{\partial z_j^{(L)}}\frac{\partial C_0}{\partial a_j^{(L)}}\end{gathered}$
- $\dots$

where:

- $\begin{gathered}C_0=\sum_{j=0}^{n_L-1}(a_j^{(L)}-y_j)^2 \end{gathered}$
	- $n_L\text–1$ parent nodes
	- $\vec y$ = desired output vector in training sample $0$
	- $\vec a^{(L)}$ = actual output vector at layer $L$
	- this is why for $\frac{\partial C_0}{\partial a_k^{(L-1)}}$ we have to iterate through multiple nodes
- $a_j^{(L)}=\sigma(\underbracket{z_j^{(L)}}) = \underbracket{ w_{j0}^{(L)}a_0^{(L-1)}}+ \cdots + \underbracket{w_{jk}^{(L)}a_k^{(L-1)}}+\cdots +b_j^{(L)}$
	- multiple parent nodes for each output node, each with their own weight
