#give me transformer model code in python to solve a sudoku where input shape is 1x81 with some empty cells and task is to fill those cells using transformers

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SudokuTransformer(nn.Module):
    def __init__(self, d_model=10, num_layers=1, num_heads=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Positional encoding layer
        self.pos_enc = nn.Parameter(self.get_positional_encoding(81, d_model), requires_grad=False)

        # Input embedding layer
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=d_model)

        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output linear layer
        self.out = nn.Linear(d_model, 10)


    def forward(self, x):
            # Input shape: (batch_size, 81)

            # convert x into long tensor
            # x = x.long()
            # Convert input to one-hot encoding

            x_one_hot = F.one_hot(x.long(), num_classes=10).float()  # Shape: (batch_size, 81, 10)
            
            # print(self.pos_enc.shape)

            # print('embedding shape', self.embedding(x_one_hot.argmax(dim=-1)).shape)
            # print('POS encoding shape', self.pos_enc[:x.shape[1], :].unsqueeze(0).shape)
            
            # print(x.shape[1])
            # Add positional encoding
            # x = self.embedding(x_one_hot.argmax(dim=-1)) + self.pos_enc[:x.shape[1], :].unsqueeze(0)
            # print(x_one_hot.shape, self.pos_enc[:x.shape[1], :].unsqueeze(0).shape)

            position = 2
            print('onecode representation', x_one_hot[0][position])
            x = x_one_hot + self.pos_enc[:x.shape[1], :].unsqueeze(0)
            print('positional encoding', self.pos_enc[:x.shape[1], :].unsqueeze(0)[0][position])
            # print(self.pos_enc[:x.shape[1], :].unsqueeze(0)[0][0])
            assert (x.dtype == torch.float32)

            # print('Positional emcoding is done  ')
            # Create mask to prevent modifying already filled cells
            mask = x_one_hot.sum(dim=-1).bool()  # Shape: (batch_size, 81)
            # print('x_one_hot', x_one_hot)
            # print('MASK', mask)
            # print('MASK shape', mask.shape)
            # Transformer encoder
            for i in range(self.num_layers):
                x = self.transformer_layers[i](x)#, src_key_padding_mask=mask)
                assert x.dtype == torch.float32
            print('learned encoding', x[0][position])
            # print('Running till here')
            # Output layer
            x = self.out(x)  # Shape: (batch_size, 81, 9)

            assert x.dtype == torch.float32

            # Mask out already filled cells
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
            assert x.dtype == torch.float32
            # v = torch.argmax(x, dim=-1)
            # print(v.shape)
            # print(x[0])
            # assert v.dtype == torch.float32
            # print(x.shape)
            return x

    def get_positional_encoding(self, length, d_model):
        # Compute positional encoding as described in the paper
        position = torch.arange(length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pos_enc = torch.zeros((length, d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc
    

import torch.optim as optim

def train(model, train_data, epochs = 10, lr = 0.001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        # print('Epoch', epoch)
        total_loss = 0
        for input, target in train_data:
            optimizer.zero_grad()
            # print(input.shape)
            # print(output.shape)
            assert input.dtype == torch.float32
            assert target.dtype == torch.float32
            output = model(input)

            assert output.dtype == torch.float32
            # #print input and ooutput shape to console in pretty format  
            # print('Input shape', input.shape)
            # print('Output shape', output.shape)
            # # output = output.type(torch.float32)
            # # print(output.grad)
            # print('TYPE, output',output.dtype)
            # # print(input.dtype)
            # print('TYPE target', target.dtype)

            target = F.one_hot(target.long(), num_classes=10).float()
            # print('output structure : ', output.shape)
            # print('taget structure : ', target.shape)

            loss = loss_fn(output.reshape(-1), target.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_data):.4f}")
    return model
        
