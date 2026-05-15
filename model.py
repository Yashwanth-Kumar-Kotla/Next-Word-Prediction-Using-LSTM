import torch.nn as nn

class LSTMNN(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 100)
        self.LSTM = nn.LSTM(100,150, batch_first = True)
        self.fc = nn.Linear(150, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        intermediate_hidden_states, (final_hidden_state, final_cell_state) = self.LSTM(embedded)
        output = self.fc(final_hidden_state.squeeze(0))
        return output