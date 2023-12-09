# Hashing and Machine Translation

## Practice Quiz

### Question 1

Assume that your objective is to minimize the transformation of X as similar to Y as possible, what would you optimize to get R? $(XR \approx Y)$

- Minimize the distance between $XR$ and $Y$
- Maximize the distance between $XR$ and $Y$
- Minimize the dot product between $XR$ and $Y$
- Maximize the dot product between $XR$ and $Y$

Answer: A

### Question 2

When solving for $R$, which of the following is true?

- Create a for loop, inside the for loop: (initialize $R$, compute the gradient, update the loss)
- Create a for loop, inside the for loop: (initialize $R$, update the loss, compute the gradient)
- Initialize $R$, create a for loop, inside the for loop:  (compute the gradient, update the loss)
- Initialize $R$, compute the gradient, create a for loop, inside the for loop:  (update the loss)

Answer: C

### Question 3

The Frobenius norm of $A=[[1,3],[4,5]]$ (Answer should be in 2 decimal places)

Answer: 7.14

### Question 4

Assume $X \in \Bbb{R}^{m\times n}$, $R \in \Bbb{R}^{n\times n}$, $Y \in \Bbb{R}^{m\times n}$, which of the following is the gradient of $\|XR - Y\|_F^2$?

- $\frac{2}{m}X^T(XR - Y)$
- $\frac{2}{m}X(XR - Y)$
- $\frac{2}{m}(XR - Y)X$
- $\frac{2}{m}(XR - Y)X^T$

Answer: A

### Question 5

Imagine that you are visiting a city in the US. If you search for friends that are living in the US, would you be able to determine the 2 closest of ALL your friends around the world?

- Yes, because I am already in the country and that implies that my closest friends are also going to be in the same country.
- No

Answer: B

### Question 6

What is the purpose of using a function to hash vectors into values?

- To speed up the time it takes when comparing similar vectors.
- To not have to spend time comparing vectors with other vectors that are completely different.
- To make the search for other similar vectors more accurate.
- It helps us create vectors.

Answer: AB

### Question 7

Given the following vectors, determine the true statements.

$P:[1,1]^T$, $V_1:[1,1]^T$, $V_2:[2,2]^T$, $V_3:[-1,-1]^T$

- $PV_1^T$ and $PV_2^T$ have the same sign.
- $PV_1^T$ and $PV_2^T$ are equal in magnitude.
- $PV_1^T$ and $PV_3^T$ have the same sign.

Answer: A

### Question 8

We define $H$ to be the number of planes and $h_i$ to be 1 or 0 depending on the sign of the dot product with plane $i$. Which of the following is the equation used to calculate the hash for several planes.

- $\sum_i^H 2^i h_i$
- $\sum_i^H 2^i h_i^i$
- $\sum_i^H 2ih_i$
- $\sum_i^H 2^{h_i} i$

Answer: A

### Question 9

How can you speed up the look up for similar documents?

- PCA
- Approximate Nearest Neighbors
- K-Means
- Locality sensitive hashing

Answer: BD

### Question 10

Hash tables are useful because

- allow us to divide vector space to regions
- speed up look up
- classify with higher accuracy
- can always be reproduced

Answer: ABD
