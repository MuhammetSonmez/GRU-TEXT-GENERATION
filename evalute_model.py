import torch
from torch import nn
import requests

device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

URL = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
response = requests.get(URL)

if response.status_code != 200:
    exit()

text = response.text
vocab:set[str] = sorted(set(text))
vocab_to_idx:dict[str, int] = {char: idx for idx, char in enumerate(vocab)}
idx_to_vocab:dict = {idx: char for char, idx in vocab_to_idx.items()}
vocab_size:int = len(vocab)
embedding_dim:int = 256
rnn_units:int = 1024


class RNNModel(nn.Module):
    def __init__(self, vocab_size:int, embedding_dim:int, rnn_units:int):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, rnn_units, batch_first=True)
        self.fc = nn.Linear(rnn_units, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden
model = RNNModel(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)
model.load_state_dict(torch.load("model.pth", weights_only=True))


def generate_text(model, start_text:str, length:int, idx_to_vocab:dict[int, str], vocab_to_idx:dict[str, int], temperature:float=1.0) -> str:
    model.eval()
    input_ids = torch.tensor([vocab_to_idx[char] for char in start_text], dtype=torch.long).unsqueeze(0).to(device)
    hidden = None

    generated_text = start_text
    print(generated_text, end="")
    for _ in range(length):
        with torch.no_grad():
            outputs, hidden = model(input_ids, hidden)
            logits = outputs[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

        next_char = idx_to_vocab[next_id]
        print(next_char, end="")

        generated_text += next_char

        input_ids = torch.tensor([[next_id]], dtype=torch.long).to(device)

    return generated_text



def generate_from_user_input():
    while True:
        user_input:str = input("")
        if user_input == "exit": break

        generate_text(model=model, start_text=user_input, length=1000, idx_to_vocab=idx_to_vocab, vocab_to_idx=vocab_to_idx, temperature=1.0)
    
generate_from_user_input()
