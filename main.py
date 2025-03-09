import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import TransformerEncoder, TransformerDecoder
from utilities import Utilities
import matplotlib.pyplot as plt

seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 16
block_size = 32
learning_rate = 1e-3
n_embd = 64
n_head = 2
n_layer = 4

eval_interval = 100
max_iters = 500
eval_iters = 200

n_input = 64
n_hidden = 100
n_output = 3
epochs_CLS = 15

class FeedForwardClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def load_texts(directory):
    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def collate_batch(batch):
    data, labels = zip(*batch)
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(encoder, classifier, data_loader):
    encoder.eval()
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            mean_emb, _ = encoder(X)  # Forward pass through encoder
            outputs = classifier(mean_emb)  # Forward pass through classifier
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
    accuracy = (100 * total_correct / total_samples)
    encoder.train()
    classifier.train()
    return accuracy

def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    decoderLMmodel.eval()
    losses = []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        
        # Only get logits from decoderLMmodel output
        logits, _ = decoderLMmodel(X)
        
        # Calculate the cross-entropy loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), reduction='mean')
        losses.append(loss.item())
        total_loss += loss.item()
        
        if len(losses) >= eval_iters:
            break

    mean_loss = torch.tensor(losses).mean()
    perplexity = torch.exp(mean_loss).item()

    decoderLMmodel.train()
    return perplexity

def main():
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts))
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)

    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
    
    # Test Dataset
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)

    print("=========================Part 1===============================")
    # Encoder
    vocab_size = tokenizer.vocab_size
    encoder = TransformerEncoder(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Number of parameters in the encoder: {num_params}")

    # Feedforward classifier
    classifier = FeedForwardClassifier(n_embd, n_hidden, n_output).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    #Sanity checking before training
    print("Sanity check before training")
    print("=============================================================")
    with torch.no_grad():  # Disable gradient tracking
        batch_iter = iter(train_CLS_loader)
        X, _ = next(batch_iter)
        sentence_1 = "It's time to put an end to micromanagement of foreign and security assistance programs—micromanagement that humiliates our friends and allies and hamstrings our diplomacy."
        utils = Utilities(tokenizer, encoder)
        utils.sanity_check(sentence_1, block_size)
    train_accuracies = []
    test_accuracies = []
    # Training 
    for epoch in range(epochs_CLS):
        total_loss = 0.0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            mean_emb, _ = encoder(xb)
            logits = classifier(mean_emb)

            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Train accuracy
        train_acc = compute_classifier_accuracy(encoder, classifier, train_CLS_loader)
        print(f"Epoch [{epoch+1}/{epochs_CLS}], Loss: {total_loss/len(train_CLS_loader):.4f}, Training Accuracy: {train_acc:.2f}%")


        # Test accuracy 
        test_acc = compute_classifier_accuracy(encoder, classifier, test_CLS_loader)
        print(f"Epoch [{epoch+1}/{epochs_CLS}], Test Accuracy: {test_acc:.2f}%")
                 # Plot the training and testing accuracy after training
    
    # Sanity checking after encoder training
    print("========================================================")
    print("Sanity check after encoder training")
    with torch.no_grad():  # Disable gradient tracking
        batch_iter = iter(train_LM_loader)
        X, _ = next(batch_iter)
        sentence_1 = "It's time to put an end to micromanagement of foreign and security assistance programs—micromanagement that humiliates our friends and allies and hamstrings our diplomacy."
        utils = Utilities(tokenizer, encoder)
        utils.sanity_check(sentence_1, block_size)
    
    # Define the decoder model, loss function, and optimizer
    decoder = TransformerDecoder(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    #Sanity checking for decoder
    print("========================================================")
    print("Sanity check after decoder training")
    batch_iter = iter(train_LM_loader)
    X, _ = next(batch_iter)
    sentence_2 = tokenizer.decode(X[0].tolist())
    utils = Utilities(tokenizer, decoder)
    utils.sanity_check(sentence_2, block_size)

    num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Number of parameters in the decoder: {num_params}")
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break

        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        logits, _ = decoder(xb)
        logits = logits.view(-1, vocab_size)  # Flatten for (batch_size * seq_len, vocab_size)
        yb = yb.view(-1)  # Flatten target for (batch_size * seq_len)
        
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        # Print loss and perplexity every 100 iterations
        if i % 100 == 0:
            perplexity = torch.exp(loss)
            print(f"Iteration {i}, Loss: {loss.item():.4f}, Perplexity: {perplexity.item():.4f}")

    # Print the final perplexity on the training set after all iterations
    final_perplexity = compute_perplexity(decoder, train_LM_loader, eval_iters=eval_iters)
    print(f"Final training set perplexity after {max_iters} iterations: {final_perplexity:.4f}")

    test_files = {
        "Obama": "speechesdataset/test_LM_obama.txt",
        "H. Bush": "speechesdataset/test_LM_hbush.txt",
        "W. Bush": "speechesdataset/test_LM_wbush.txt"
    }

    for politician, file_path in test_files.items():
        with open(file_path, 'r', encoding='utf-8') as f:
            test_text = f.read()

        test_LM_dataset = LanguageModelingDataset(tokenizer, test_text, block_size)
        test_LM_loader = DataLoader(test_LM_dataset, batch_size=batch_size, shuffle=False)

        # Compute perplexity for each politician's test set
        perplexity = compute_perplexity(decoder, test_LM_loader, eval_iters=eval_iters)
        print(f"Perplexity for {politician} test set: {perplexity:.4f}")


if __name__ == "__main__":
    main()

