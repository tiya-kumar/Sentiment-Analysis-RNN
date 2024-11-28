import numpy as np
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



def tokenize(text):
    # Lowercase
    text = text.lower()
    # Remove all punc except for apostrophes
    text = re.sub(r"[^\w\d'\s]+", '', text)
    # Split on whitespace
    tokens = text.split()
    return tokens


def create_vocab(words):
    vocab = set()
    for sentence in sentences:
        tokens = tokenize(sentence)
        vocab.update(tokens)
    return vocab


def encode_sentences(sentences, vocabulary):
    word_to_index = {word: i for i, word in enumerate(vocabulary)}
    encoded_sentences = []
    for sentence in sentences:
        tokens = tokenize(sentence)
        encoded_sentence = [word_to_index[word] for word in tokens]
        encoded_sentences.append(encoded_sentence)
    return encoded_sentences, word_to_index


def pad_sequences(sequences, max_length):
    padded_sequences = []
    for sequence in sequences:
        if len(sequence) < max_length:
            padded_sequence = sequence + [0] * (max_length - len(sequence))
        else:
            padded_sequence = sequence[:max_length]
        padded_sequences.append(padded_sequence)
    return padded_sequences

# Done with functions for tokenizing and padding at this point

# sentences = [
#     "I am overjoyed with the results!",
#     "This news broke my heart.",
#     "Today was an ordinary day.",
#     "I feel indifferent towards this movie",
#     "I am grateful for all the support I have received.",
#     "This situation has left me feeling frustrated."
# ]
# sentiment_labels = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0]]

# # Tokenize each sentence
# tokenized_sentences = [tokenize(sentence) for sentence in sentences]
# # Create vocab
# vocabulary = create_vocab(sentences)

# # Encode (replace word with corresponding index)
# encoded_sentences, word_to_index = encode_sentences(sentences, vocabulary)

# # Find maximum length
# max_length_result = max(len(sentence) for sentence in encoded_sentences)

# # Pad sequences to ensure length is the same
# padded_sequences_result = pad_sequences(encoded_sentences, max_length_result)


# # Splitting data set into training and testing...right now it is 4:2 remember to add more later
# X_train, X_test, y_train, y_test = train_test_split(padded_sequences_result, sentiment_labels, test_size=0.2, random_state=42)


# FINALLY begin building recurrent net YAYYY 


# make sure all words in testing set have to occur in the training set
class Node(object):
    def __init__(self, id, kind):
        self.id = id
        self.kind = kind
        self.activation = 0.0
        self.desired_activation = 0.0
        self.net_input = 0.0
        self.raw_error = 0.0
        self.corrected_error = 0.0
        self.error_contribution = 0.0
        self.incoming_connections = []
        self.outgoing_connections = []

    def add_incoming_connection(self, sender_node):
        # adding incoming connections
        connection = Connection(recipient=self, sender=sender_node)
        self.incoming_connections.append(connection)
        sender_node.outgoing_connections.append(connection)

    def add_outgoing_connection(self, recipient_node):
        # outogoing connctions
        connection = Connection(sender=self, recipient=recipient_node)
        self.outgoing_connections.append(connection)
        recipient_node.incoming_connections.append(connection)

    def activate(self):
        self.activation = sigmoid(sum(conn.weight * conn.sender.activation for conn in self.incoming_connections))

    # calculating errors under here
    def calculate_output_error(self):
        self.raw_error = self.desired_activation - self.activation
        self.corrected_error = self.raw_error * sigmoid_derivative(self.activation)

    def calculate_hidden_error(self):
        self.error_contribution = sum(conn.weight * conn.recipient.raw_error for conn in self.outgoing_connections)
        self.corrected_error = self.error_contribution * sigmoid_derivative(self.activation)

    def update_weights(self, learning_rate):
        for conn in self.incoming_connections:
            conn.weight += learning_rate * conn.sender.activation * self.corrected_error

    def reset_activation(self):
        self.activation = 0.0

class Connection(object):
    def __init__(self, recipient=None, sender=None, weight=None):
        self.weight = np.random.randn() * 0.01 if weight is None else weight
        self.change_in_weight = 0.0
        self.recipient = recipient
        self.sender = sender
# define sigmoid function AND DERIVATIVE
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def build_srn(input_size, hidden_size, output_size):
    input_nodes = [Node(id=i, kind='input') for i in range(input_size)]
    hidden_nodes = [Node(id=i + input_size, kind='hidden') for i in range(hidden_size)]
    output_nodes = [Node(id=i + input_size + hidden_size, kind='output') for i in range(output_size)]
    # connecting each layer to each other
    for input_node in input_nodes:
        for hidden_node in hidden_nodes:
            input_node.add_outgoing_connection(hidden_node)

    for hidden_node in hidden_nodes:
        for output_node in output_nodes:
            hidden_node.add_outgoing_connection(output_node)

    for hidden_node in hidden_nodes:
        hidden_node.add_incoming_connection(hidden_node)

    return input_nodes, hidden_nodes, output_nodes

def train_srn(epochs, learning_rate, input_nodes, hidden_nodes, output_nodes, X, y):
    for epoch in range(epochs):
        for i in range(len(X)):
            for node in hidden_nodes + output_nodes:
                node.reset_activation()

            for t in range(X.shape[1]):
                for idx, input_node in enumerate(input_nodes):
                    input_node.activation = 1 if idx == X[i, t] else 0

                for node in hidden_nodes + output_nodes:
                    node.activate()

                for idx, output_node in enumerate(output_nodes):
                    #output_node.desired_activation = y[i][idx] if y[i][idx] else 0
                    output_node.desired_activation = y[i, idx]
                    output_node.calculate_output_error()
                    output_node.update_weights(learning_rate)

                for node in hidden_nodes:
                    node.calculate_hidden_error()
                    node.update_weights(learning_rate)


def get_activations(nodes, data):
    activations = []
    for seq in data:
        for node in nodes:
            node.reset_activation()
        for i in range(len(seq)):
            for idx, node in enumerate(nodes):
                node.activation = 1 if seq[i] == idx else 0
            for node in nodes:
                node.activate()
        activations.append([node.activation for node in nodes])
    return np.array(activations)

#network = build_srn(input_size=vocabulary, hidden_size=50, output_size=vocab_size)
#train_srn(epochs=10, learning_rate=0.01, input_nodes=network[0], hidden_nodes=network[1], output_nodes=network[2], X=X, y=y)
#probe_sequences = tokenizer.texts_to_sequences(probe_words)
#probe_data = pad_sequences(probe_sequences, maxlen=max_length, padding='post')
#probe_activations = get_activations(network[1], probe_data)
#tsne_results = TSNE(n_components=2, perplexity=30, n_iter=3000).fit_transform(probe_activations)
#colors = [probe_category_map[word] for word in probe_words]
#plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, cmap='viridis', alpha=0.6)
#plt.colorbar()
#plt.show()

sentences = [
    "I am overjoyed with the results!",
    "This news broke my heart.",
    "I am grateful for all the support I have received.",
    "This situation has left me feeling frustrated.",
    "Today was an ordinary day.",
    "I feel indifferent towards this movie"
]
sentiment_labels = [[0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0]]

# Tokenize each sentence
tokenized_sentences = [tokenize(sentence) for sentence in sentences]
# Create vocab
vocabulary = create_vocab(sentences)
print("Vocabulary:", vocabulary)

# Encode (replace word with corresponding index)
encoded_sentences, word_to_index = encode_sentences(sentences, vocabulary)

# Find maximum length
max_length_result = max(len(sentence) for sentence in encoded_sentences)

# Pad sequences to ensure length is the same
padded_sequences_result = pad_sequences(encoded_sentences, max_length_result)
print("\nRandom Padded sequences:")
for sentence, padded_sequence in zip(sentences, padded_sequences_result):
    print(sentence + " ->", padded_sequence)

# Splitting data set into training and testing...right now it is 4:2 remember to add more later
X_train, X_test, y_train, y_test = train_test_split(np.array(padded_sequences_result), np.array(sentiment_labels), test_size=0.2, random_state=42)

print("\nTraining set size:", len(X_train))
print("Testing set size:", len(X_test))

# Build the SRN
input_nodes, hidden_nodes, output_nodes = build_srn(input_size=max_length_result, hidden_size=16, output_size=len(sentiment_labels[0]))

# Train the SRN
train_srn(epochs=10, learning_rate=0.1, input_nodes=input_nodes, hidden_nodes=hidden_nodes, output_nodes=output_nodes, X=X_train, y=y_train)

# Get activations for testing data
test_activations = get_activations(nodes=output_nodes, data=X_test)

# Print test activations
print("\nTest activations:")
print(test_activations)

# Get activations for testing data
test_activations = get_activations(nodes=output_nodes, data=X_test)

# Print sentiment analysis results
print("\nSentiment Analysis Results:")
for i, activations in enumerate(test_activations):
    # Find the index of the output node with the highest activation
    predicted_class = np.argmax(activations)
    
    # Map the index to the corresponding sentiment label
    if predicted_class == 0:
        sentiment = "Negative"
    elif predicted_class == 1:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"
    
    print(f"Sample {i + 1}: Predicted Sentiment : {sentiment}")
