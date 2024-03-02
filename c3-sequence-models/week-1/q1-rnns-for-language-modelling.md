# RNNs for Language Modelling

## Practice Quiz

### Question 1

For the embedding layer in your model, you'd have to learn a matrix of weights of what size?

- Equal to your vocabulary times the dimension of the number of layers
- Equal to your vocabulary times the dimension of the embedding
- Equal to the dimension of the embedding times the first dimension of the matrix in the first layer
- Equal to your vocabulary times the dimension of the number of classes

Answer: B

### Question 2

What would be the probability of a five word sequence using a penta-gram?

- $P(w_5 \vert w_4, w_3, w_2, w_1) = \frac{count(w_5, w_4, w_3, w_2, w_1)}{count(w_4, w_3, w_2, w_1)}$
- $P(w_5, w_4, w_3, w_2, w_1) = P(w_1) \times P(w_2 \vert w_1) \times P(w_3 \vert w_1, w_2) \times P(w_4 \vert w_1, w_2, w_3) \times P(w_5 \vert w_1, w_2, w_3, w_4)$
- $P(w_5, w_4, w_3, w_2, w_1) = P(w_1) \times P(w_2) \times P(w_3) \times P(w_4) \times P(w_5)$
- $P(w_5, w_4, w_3, w_2, w_1) = P(w_5 \vert w_4, w_3, w_2, w_1)$

Answer: B

### Question 3

The number of parameters in an RNN is the same regardless of the input's length.

- True
- False

Answer: A

### Question 4

Select all the examples that correspond to a "many to one" architecture.

- An RNN which inputs a sentiment and generates a sentence.
- An RNN which inputs a sentence and determines the sentiment.
- An RNN which inputs a topic and generates a conversation about that topic.
- An RNN which inputs a conversation and determines the topic.

Answer: BD

### Question 5

What should be the size of matrix $W_h$, if $h^{<t>}$ had size $4 \times 1$ and $x^{<t>}$ $10 \times 1$?

- $4 \times 14$
- $14 \times 4$
- $4 \times 4$
- $14 \times 14$

Answer: A

### Question 6

In the next equation, why is there a division by the number of time steps but not one for the number of classification categories?

$J = -\frac{1}{T} \sum_{t=1}^T \sum_{j=1}^K y_j^{<t>} \log \hat{y}_j^{<t>}$

- Because there is just one value in every vector $y^{<t>}$ different from zero.
- Because the equation is wrong.
- Because this equation is given for a single example.
- Because for most classification tasks there are only two categories.

Answer: A

### Question 7

What problem, related to vanilla RNNs, do GRUs tackle?

- Loss of relevant information for long sequences of words.
- Overfitting
- High computational time for training and prediction.
- Restricted flow of information from the past to the present.

Answer: A

### Question 8

Bidirectional RNNs are acyclic graphs, which means that the computations in one direction are independent from the ones in the other direction.

- True
- False

Answer: A

### Question 9

Compared to Traditional Language models which of the following problems does an RNN help us with?

- Helps us solve RAM issues.
- Helps us solve memory issues.
- They are much simpler to understand.
- They require almost no knowledge to use when compared to the traditional n-gram model.

Answer: AB

### Question 10

What type of RNN structure would you use when implementing machine translation?

- One to many
- Many to one
- One to one
- Many to Many

Answer: D
