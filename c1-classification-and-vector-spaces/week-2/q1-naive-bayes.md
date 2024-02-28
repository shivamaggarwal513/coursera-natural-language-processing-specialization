# Naive Bayes

## Practice Quiz

### Question 1

Assume that there are 2 happy people and 2 unhappy people in a room. Concretely, persons A and B are happy and persons C and D are unhappy. If you were to randomly pick a person from the room, what is the probability that the person is happy.

- $1/2$
- $1/4$
- $3/4$
- $0$

Answer: A

### Question 2

Assume that there are 2 happy people and 2 unhappy people in a room. Concretely, persons A and B are happy and persons C and D are unhappy. If a friend showed you the part of the room where the two happy people are, what is the probability that you choose person B?

- $1/2$
- $1/4$
- $3/4$
- $1$

Answer: A

### Question 3

From the equations presented below, express the probability of a tweet being positive given that it contains the word happy in terms of the probability of a tweet containing the word happy given that it is positive
$$P\left(\text{Positive | "happy"}\right) = \frac{P\left(\text{Positive} \cap \text{"happy"}\right)}{P\left(\text{"happy"}\right)}$$
$$P\left(\text{"happy" | Positive}\right) = \frac{P\left(\text{"happy"} \cap \text{Positive}\right)}{P\left(\text{Positive}\right)}$$

- $P\left(\text{Positive | "happy"}\right) = P\left(\text{"happy" | Positive}\right) \times \frac{P\left(\text{Positive}\right)}{P\left(\text{"happy"}\right)}$
- $P\left(\text{Positive | "happy"}\right) = P\left(\text{"happy" | Positive}\right) \times \frac{P\left(\text{"happy"}\right)}{P\left(\text{Positive}\right)}$
- $P\left(\text{Positive} \cap \text{"happy"}\right) = P\left(\text{"happy" | Positive}\right) \times \frac{P\left(\text{Positive}\right)}{P\left(\text{"happy"}\right)}$
- $P\left(\text{Positive} \cap \text{"happy"}\right) = P\left(\text{"happy" | Positive}\right) \times \frac{P\left(\text{"happy"}\right)}{P\left(\text{Positive}\right)}$

Answer: A

### Question 4

Bayes rule is defined as

- $P\left(X|Y\right) = P\left(Y|X\right)\times\frac{P(X)}{P(Y)}$
- $P\left(X|Y\right) = P\left(Y|X\right)\times\frac{P(Y)}{P(X)}$
- $P\left(X|Y\right) = P\left(X|Y\right)\times\frac{P(X)}{P(Y)}$
- $P\left(X|Y\right) = P\left(Y|X\right)\times\frac{P(X)}{P(Y|X)}$

Answer: A

### Question 5

Suppose that in your dataset, 25% of the positive tweets contain the word 'happy'. You also know that a total of 13% of the tweets in your dataset contain the word 'happy', and that 40% of the total number of tweets are positive. You observe the tweet: "happy to learn NLP". What is the probability that this tweet is positive? (Please, round your answer up to two decimal places. Remember that 0.578 = 0.58 and 0.572 = 0.57)

Answer: 0.77

### Question 6

The log likelihood for a certain word $w_i$ is defined as $\log\left(\frac{P(w_i|pos)}{P(w_i|neg)}\right)$

- Positive numbers imply that the word is positive.
- Positive numbers imply that the word is negative.
- Negative numbers imply that the word is negative.
- Negative numbers imply that the word is positive.

Answer: AC

### Question 7

The log likelihood mentioned in lecture, which is the log of the ratio between two probabilities is bounded between

- $-1$ and $1$
- $-\infty$ and $\infty$
- $0$ and $\infty$
- $0$ and $1$

Answer: B

### Question 8

When implementing naive Bayes, in which order should the following steps be implemented.

- 1. Get or annotate a dataset with positive and negative tweets
  1. Preprocess the tweets: `process_tweet(tweet)`
  1. Compute `freq(w, class)`
  1. Get $P(w|pos)$, $P(w|neg)$
  1. Get $\lambda(w)$
  1. Compute $logprior = \log\left(\frac{P(pos)}{P(neg)}\right)$

- 1. Get or annotate a dataset with positive and negative tweets
  1. Preprocess the tweets: `process_tweet(tweet)`
  1. Compute `freq(w, class)`
  1. Get $\lambda(w)$
  1. Get $P(w|pos)$, $P(w|neg)$
  1. Compute $logprior = \log\left(\frac{P(pos)}{P(neg)}\right)$

- 1. Get or annotate a dataset with positive and negative tweets
  1. Compute `freq(w, class)`
  1. Preprocess the tweets: `process_tweet(tweet)`
  1. Get $P(w|pos)$, $P(w|neg)$
  1. Get $\lambda(w)$
  1. Compute $logprior = \log\left(\frac{P(pos)}{P(neg)}\right)$

- 1. Get or annotate a dataset with positive and negative tweets
  1. Compute `freq(w, class)`
  1. Preprocess the tweets: `process_tweet(tweet)`
  1. Compute $logprior = \log\left(\frac{P(pos)}{P(neg)}\right)$
  1. Get $P(w|pos)$, $P(w|neg)$
  1. Get $\lambda(w)$

Answer: A

### Question 9

To test naive bayes model, which of the following are required?

- $X_{val}, Y_{val}, \lambda, logprior$
- $X_{val}, Y_{val}, logprior$
- $X_{val}, \lambda, logprior$
- $Y_{val}, \lambda, logprior$

Answer: A

### Question 10

Which of the following is NOT an application of Naive Bayes?

- Sentiment Analysis
- Author identification
- Information retrieval
- Word disambiguation
- Numerical predictions

Answer: E
