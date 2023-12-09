# Vector Space Models

## Practice Quiz

### Question 1

Given a corpus A, encoded as $[1, 2, 3]^T$ and corpus B encoded as $[4, 7, 2]^T$. What is the euclidean distance between the two documents?

- $5.91608$
- $35$
- $2.43$
- None of the above

Answer: A

### Question 2

Given the previous problem, a user now came up with a corpus C defined as $[3, 1, 4]^T$ and you want to recommend a document that is similar to it. Would you recommend document A or document B?

- Document A
- Document B

Answer: A

### Question 3

Which of the following is true about euclidean distance?

- When comparing similarity between two corpuses, it does not work well when the documents are of different sizes.
- It is the norm of the difference between two vectors.
- It is a method that makes use of the angle between two vectors
- It is the norm squared of the difference between two vectors.

Answer: AB

### Question 4

What is the range of a cosine similarity score, namely $s$, in the case of information retrieval where the vectors are positive?

- $-1 \le s \le 1$
- $-\infty \le s \le \infty$
- $0 \le s \le 1$
- $-1 \le s \le 0$

Answer: C

### Question 5

The cosine similarity score of corpus A = $[1, 0, -1]^T$ and corpus B = $[2, 8, 1]^T$ is equal to

- $0.08512565307587486$
- $0$
- $1.251903$
- $-0.3418283$

Answer: A

### Question 6

We will define the following vectors, USA = $[5, 6]^T$, Washington = $[10, 5]^T$, Turkey = $[3, 1]^T$, Ankara = $[9, 1]^T$, Russia = $[5, 5]^T$, and Japan = $[4, 3]^T$. Using only the following vectors, Ankara is the capital of what country? Please consider the cosine similarity score in your calculations.

- Japan
- Russia
- Morocco
- Turkey

Answer: D

### Question 7

Please select all that apply. PCA is

- used to reduce the dimension of your data
- visualize word vectors
- make predictions
- label data

Answer: AB

### Question 8

Please select all that apply. Which is correct about PCA?

- You can think of an eigenvector as an uncorrelated feature for your data.
- The eigenvalues tell you the amount of information retained by each feature.
- If working with features in different scales, you do not have to mean normalize.
- Computing the covariance matrix is critical when performing PCA.

Answer: ABD

### Question 9

In which order do you perform the following operations when computing PCA?

- mean normalize, get $\sum$ the covariance matrix, perform SVD, then dot product the data, namely X, with a subset of the columns of U to get the reconstruction of your data.
- mean normalize, perform SVD, get $\sum$ the covariance matrix, then dot product the data, namely X, with a subset of the columns of U to get the reconstruction of your data.
- get $\sum$ the covariance matrix, perform SVD, then dot product the data, namely X, with a subset of the columns of U to get the reconstruction of your data, mean normalize.
- get $\sum$ the covariance matrix, mean normalize, perform SVD, then dot product the data, namely X, with a subset of the columns of U to get the reconstruction of your data.

Answer: A

### Question 10

Vector space models allow us to

- to represent words and documents as vectors.
- build useful applications including and not limited to, information extraction, machine translation, and chatbots.
- create representations that capture similar meaning.
- build faster training algorithms.

Answer: ABC
