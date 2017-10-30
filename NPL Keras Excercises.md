
### Data preparation


```python
from keras.preprocessing.text import text_to_word_sequence, one_hot, hashing_trick
# define the document
text = 'The quick brown fox jumped over the lazy dog.'
# tokenize the document
result = text_to_word_sequence(text)
print(result)
```

    ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
    


```python
print(result)
```

    ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
    


```python
unique_words = set(result)   # get the list of unique words
length_words = len(unique_words)
print("Unique Words:", unique_words, "  and the numbers of unique words:", length_words)
```

    Unique Words: {'dog', 'brown', 'over', 'lazy', 'quick', 'the', 'jumped', 'fox'}   and the numbers of unique words: 8
    


```python
text_encoded = one_hot(text, round(length_words*13.3))
print(text_encoded)
```

    [91, 44, 95, 30, 66, 65, 91, 65, 1]
    


```python
result = hashing_trick(text, round(length_words*1.3), hash_function='md5')
print(result)
```

    [6, 4, 1, 2, 7, 5, 6, 2, 6]
    


```python
from keras.preprocessing.text import Tokenizer
# define 5 documents
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!']
# create the tokenizer
t = Tokenizer()           # Tokenizer is a class
# fit the tokenizer on the documents
t.fit_on_texts(docs)      # fit_on_texts is a method of Tokenizer class
```

Once fit, the Tokenizer provides 4 attributes that you can use to query what has been learned about your documents:

word_counts: A dictionary of words and their counts.
word_docs: An integer count of the total number of documents that were used to fit the Tokenizer. 
word_index: A dictionary of words and their uniquely assigned integers. 
document_count: A dictionary of words and how many documents each appeared in. 

Once the Tokenizer has been fit on training data, it can be used to encode documents in the train or test datasets.


```python
# summarize what was learned
print("Word Count:", t.word_counts)
print("Document Count:", t.document_count)
print("Word Index:", t.word_index)
print("Document Count:", t.word_docs)
```

    Word Count: OrderedDict([('well', 1), ('done', 1), ('good', 1), ('work', 2), ('great', 1), ('effort', 1), ('nice', 1), ('excellent', 1)])
    Document Count: 5
    Word Index: {'well': 2, 'good': 4, 'nice': 7, 'work': 1, 'effort': 6, 'done': 3, 'excellent': 8, 'great': 5}
    Document Count: {'well': 1, 'good': 1, 'nice': 1, 'work': 2, 'great': 1, 'done': 1, 'excellent': 1, 'effort': 1}
    

The texts_to_matrix() function on the Tokenizer can be used to create one vector per document provided per input. The length of the vectors is the total size of the vocabulary.

This function provides a suite of standard bag-of-words model text encoding schemes that can be provided via a mode argument to the function.

The modes available include:

'binary': Whether or not each word is present in the document. This is the default.
'count': The count of each word in the document.
'tfidf': The Text Frequency-Inverse DocumentFrequency (TF-IDF) scoring for each word in the document.
'freq': The frequency of each word as a ratio of words within each document.


```python
from keras.preprocessing.text import Tokenizer
# integer encode documents
encoded_docs = t.texts_to_matrix(docs, mode='count')        # texts_to_matrix is a method of Tokenizer class
print(encoded_docs)
```

    [[ 0.  0.  1.  1.  0.  0.  0.  0.  0.]
     [ 0.  1.  0.  0.  1.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  1.  1.  0.  0.]
     [ 0.  1.  0.  0.  0.  0.  0.  1.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  1.]]
    

### Embeddings


```python
# define documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']
# define class labels
labels = [1,1,1,1,1,0,0,0,0,0]           # "1" for positive comment, "0" for negative comment
```

Next, we can integer encode each document. This means that as input the Embedding layer will have sequences of integers. 

Keras provides the one_hot() function that creates a hash of each word as an efficient integer encoding. We will estimate the vocabulary size of 50, which is much larger than needed to reduce the probability of collisions from the hash function.

We could also experiment with other more sophisticated bag of word model encoding like counts or TF-IDF.


```python
# integer encode the documents using one_hot()
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]        # this will be the input to Embedding layer
print(encoded_docs)
```

    [[6, 37], [49, 12], [30, 13], [11, 12], [45], [12], [24, 13], [25, 49], [24, 12], [23, 34, 37, 13]]
    

The sequences have different lengths and Keras prefers inputs to be vectorized and all inputs to have the same length. We will pad all input sequences to have the length of 4. Again, we can do this with a built in Keras function, in this case the pad_sequences() function.


```python
# pad documents to a max length of 4 words
from keras.preprocessing.sequence import pad_sequences
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
```

    [[ 6 37  0  0]
     [49 12  0  0]
     [30 13  0  0]
     [11 12  0  0]
     [45  0  0  0]
     [12  0  0  0]
     [24 13  0  0]
     [25 49  0  0]
     [24 12  0  0]
     [23 34 37 13]]
    


```python
# integer encode the documents using t.texts_to_sequences()
# create the tokenizer
t = Tokenizer()           # Tokenizer is a class
# fit the tokenizer on the documents
t.fit_on_texts(docs)      # fit_on_texts is a method of Tokenizer class
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)
```

    [[6, 2], [3, 1], [7, 4], [8, 1], [9], [10], [5, 4], [11, 3], [5, 1], [12, 13, 2, 14]]
    


```python
# pad documents to a max length of 4 words
vocab_size = 50
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
```

    [[ 6  2  0  0]
     [ 3  1  0  0]
     [ 7  4  0  0]
     [ 8  1  0  0]
     [ 9  0  0  0]
     [10  0  0  0]
     [ 5  4  0  0]
     [11  3  0  0]
     [ 5  1  0  0]
     [12 13  2 14]]
    

We are now ready to define our Embedding layer as part of our neural network model.

The Embedding has a vocabulary of 50 and an input length of 4. We will choose a small embedding space of 8 dimensions.

The model is a simple binary classification model. Importantly, the output from the Embedding layer will be 4 vectors of 8 dimensions each, one for each word. We flatten this to a one 32-element vector to pass on to the Dense output layer.


```python
# define the model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

model = Sequential()
doc_embeddings = Embedding(vocab_size, 8, input_length=max_length) # vocab_size = 50, max_length = 4 is max document size (number of words in each document)
print("Output of Embedding:", doc_embeddings)

```

    Output of Embedding: <keras.layers.embeddings.Embedding object at 0x00000000100D0160>
    


```python
model.add(doc_embeddings)   
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_8 (Embedding)      (None, 4, 8)              400       
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 32)                0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 33        
    =================================================================
    Total params: 433
    Trainable params: 433
    Non-trainable params: 0
    _________________________________________________________________
    None
    


```python
# fit the model
model.fit(padded_docs, labels, epochs=150, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))
```

    Accuracy: 100.000000
    

### Using Pre-Trained GloVe Embedding

 Load the entire GloVe word embedding file into memory as a dictionary of word to embedding array, which is pretty slow. It might be better to filter the embedding for the unique words in your training data.
 If you peek inside the file, you will see a token (word) followed by the weights (100 numbers) on each line. For example, below are the first line of the embedding ASCII text file showing the embedding for "the":

the -0.038194 -0.24487 0.72812 -0.39961 0.083172 0.043953 -0.39141 0.3344 -0.57545 0.087459 0.28787 -0.06731 0.30906 -0.26384 -0.13231 -0.20757 0.33395 -0.33848 -0.31743 -0.48336 0.1464 -0.37304 0.34577 0.052041 0.44946 -0.46971 0.02628 -0.54155 -0.15518 -0.14107 -0.039722 0.28277 0.14393 0.23464 -0.31021 0.086173 0.20397 0.52624 0.17164 -0.082378 -0.71787 -0.41531 0.20335 -0.12763 0.41367 0.55187 0.57908 -0.33477 -0.36559 -0.54857 -0.062892 0.26584 0.30205 0.99775 -0.80481 -3.0243 0.01254 -0.36942 2.2167 0.72201 -0.24978 0.92136 0.034514 0.46745 1.1079 -0.19358 -0.074575 0.23353 -0.052062 -0.22044 0.057162 -0.15806 -0.30798 -0.41625 0.37972 0.15006 -0.53212 -0.2055 -1.2526 0.071624 0.70565 0.49744 -0.42063 0.26148 -1.538 -0.30223 -0.073438 -0.28312 0.37104 -0.25217 0.016215 -0.017099 -0.38984 0.87424 -0.72569 -0.51058 -0.52028 -0.1459 0.8278 0.27062


```python
# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]           # this is the word, e.g., "the"
	coefs = asarray(values[1:], dtype='float32')     # the rest are the coefficients or weights
	embeddings_index[word] = coefs                   # word is used as the index to the weights
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
```

Next, we need to create a matrix of one embedding for each word in the training dataset. We can do that by enumerating all unique words in the Tokenizer.word_index and locating the embedding weight vector from the loaded GloVe embedding.

The result is a matrix of weights only for words we will see during training.


```python
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))         # vocab_size is the number of unique words in the training documents
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)   # this gets us the embedding (weights) of each unique word in the training documents
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector      # i is important because it is the index of each word and it binds the words with the weights
```

Now we can define our model, fit, and evaluate it as before.

The key difference is that the embedding layer can be seeded with the GloVe word embedding weights. We chose the 100-dimensional version, therefore the Embedding layer must be defined with output_dim set to 100. Finally, we do not want to update the learned word weights in this model, therefore we will set the trainable attribute for the model to be False.


```python
# define model
model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# summarize the model
print(model.summary())

# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)

# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))
```
