# LSTMs and Named Entity Recognition

## Practice Quiz

### Question 1

Identify the correct order of the gates that information flows through in an LSTM unit.

- Input gate, forget gate, output gate.
- Forget gate, input gate, output gate.
- Output gate, forget gate, input gate.
- Forget gate, output gate, input gate

Answer: B

### Question 2

Which are some applications of LSTMs?

- Next character prediction
- Image captioning
- Chatbots
- Music composition
- Speech recognition

Answer: ABCDE

### Question 3

The tanh layer ensures the values in your network stay numerically stable, by squeezing all values between -1 and 1. This prevents any of the values from the current inputs from becoming so large that they make the other values insignificant.

- True
- False

Answer: A

### Question 4

What type of architecture is a named entity recognition using?

- Many to many
- One to many
- Many to one

Answer: A

### Question 5

Extract the named entities from the following sentence:

Younes, a Moroccan artificial intelligence engineer, travelled to France for a conference.

- Younes, Moroccan engineer, France.
- Younes, Moroccan, conference.
- Younes, Moroccan, France.
- Younes, Moroccan, engineer.

Answer: C

### Question 6

In a vectorized representation of your data, equal sequence length allows more efficient batch processing.

- True
- False

Answer: A

### Question 7

Why is it important to mask padded tokens when computing the loss?

- Padded tokens are not part of the data and are just used to help us keep the same sequence length for more efficient batch processing. We should not include their loss.
- We add the loss of the padded tokens independently.

Answer: A

### Question 8

In which of the following orders should we train an Named Entity Recognition with an LSTM?

- 1. Create a tensor for each input and its corresponding number
  1. Put them in a batch =>  64, 128, 256, 512 ...
  1. Run the output through a dense layer
  1. Predict using a log softmax over K classes
  1. Feed it into an LSTM unit
- 1. Create a tensor for each input and its corresponding number
  1. Put them in a batch =>  64, 128, 256, 512 ...
  1. Feed it into an LSTM unit
  1. Run the output through a dense layer
  1. Predict using a log softmax over K classes
- 1. Create a tensor for each input and its corresponding number
  1. Put them in a batch =>  64, 128, 256, 512 ...
  1. Run the output through a dense layer
  1. Feed it into an LSTM unit
  1. Predict using a log softmax over K classes

Answer: B

### Question 9

LSTMs solve vanishing/exploding gradient problems when compared to basic RNNs.  

- True
- False

Answer: A

### Question 10

Which of the following are true about LSTMs and vanilla RNNs?

- LSTMs are typically trained faster than vanilla RNNs.
- LSTMs can better retain information from earlier parts of the sentence.
- LSTMs suffer from vanishing gradients, but RNNs don't.
- LSTMs suffer from exploding gradients, but RNNs don't.
- A single LSTM cell is more complex than a single cell in vanilla RNN.

Answer: BE

Explanation: LSTMs use input, output and forget gates to propagate information in a more sophisticated way than vanilla RNNs.
