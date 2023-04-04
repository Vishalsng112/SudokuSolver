#give me transformer model code in python to solve a sudoku where input shape is 1x81 with some empty cells and task is to fill those cells using transformers

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
class SudokuTransformer(nn.Module):
    def __init__(self, d_model=10, num_layers=6, num_heads=1, dropout=0.0):
        """
        d_model: dimension of the model
        num_layers: number of transformer encoder layers
        num_heads: number of heads in multi-head attention
        dropout: dropout rate
        """
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
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first = True, norm_first = True, dim_feedforward = d_model*4, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output linear layer
        self.out = nn.Linear(d_model, 10)

        # self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        # self.linear = nn.Linear(d_model, 10)



    def forward(self, x, attention_mask=None):
            # Input shape: (batch_size, 81)
            
            #get one hot encoding of input
            x_one_hot = F.one_hot(x.long(), num_classes=10).float()  # Shape: (batch_size, 81, 10)
            # print('x_one_hot', x_one_hot.shape)
            # print(self.pos_enc.shape)

            # print('embedding shape', self.embedding(x_one_hot.argmax(dim=-1)).shape)
            # print('POS encoding shape', self.pos_enc[:x.shape[1], :].unsqueeze(0).shape)
            
            # print(x.shape[1])
            # Add positional encoding
            # x = self.embedding(x_one_hot.argmax(dim=-1)) + self.pos_enc[:x.shape[1], :].unsqueeze(0)
            # print(x_one_hot.shape, self.pos_enc[:x.shape[1], :].unsqueeze(0).shape)

            # position = 2
            # print('onecode representation', x_one_hot[0][position])
            x = x_one_hot + self.pos_enc[:x.shape[1], :].unsqueeze(0)

            # # print('x', x.shape)
            # b = x.shape[0]
            # s = x.shape[1]
            # d = x.shape[2]
            # norm = torch.linalg.norm(x.view(b*s, d), dim=-1)
            # norm = norm.view(b, s, 1)
            # x = x/norm

            # print('x', x.shape)
            # print('positional encoding', self.pos_enc[:x.shape[1], :].unsqueeze(0)[0][position])
            # print(self.pos_enc[:x.shape[1], :].unsqueeze(0)[0][0])
            # assert (x.dtype == torch.float32)

            # print('Positional emcoding is done  ')
            # Create mask to prevent modifying already filled cells
            # print(x_one_hot.shape)
            # mask = x_one_hot.sum(dim=-1).bool()  # Shape: (batch_size, 81)
            # fill mask with 1s where endocing on digit 0 is present
            #create a mask of 0s
            # mask = torch.zeros(x_one_hot.shape[0], x_one_hot.shape[1], dtype=torch.bool)
            # mask = mask |(x_one_hot[:, :, 0] == 1)
            
            # print(mask)
            # #permute mask to match the shape of x
            # mask = mask.permute(1, 0)
            # print(mask.shape)
            # # Transformer encoder

            # print('X embedding', x.shape)
            # print(mask.shape)
            for i in range(self.num_layers):
                # x = self.transformer_layers[i](x, src_key_padding_mask=mask)
                x = self.transformer_layers[i](x)#, src_key_padding_mask= ~attention_mask)
                # assert x.dtype == torch.float32
            # print('learned encoding', x[0][position])
            # print('Running till here')
            # Output layer
            x = self.out(x)  # Shape: (batch_size, 81, 9)
            # assert x.dtype == torch.float32

            # # Mask out already filled cells
            # x = x.masked_fill(mask.unsqueeze(-1), 0.0)
            # assert x.dtype == torch.float32
            # # v = torch.argmax(x, dim=-1)
            # # print(v.shape)
            # # print(x[0])
            # # assert v.dtype == torch.float32
            # print(x.shape)

            #apply softmax to get probability distribution
            # x = F.softmax(x, dim=-1)
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
    #use MSE as loss function
    # loss_fn = nn.MSELoss()
    epoch_loss = []
    for epoch in range(epochs):
        # print('Epoch', epoch)
        total_loss = 0
        for input, target in train_data:

            #compute attention mask for input
            attention_mask = (input != 0).bool()
            # print(input.shape)
            # print(attention_mask.shape)
            # print(attention_mask)
            # print(input.shape)
            # print(output.shape)
            # assert input.dtype == torch.float32
            # assert target.dtype == torch.float32

            # print(input.shape)
            output = model(input)#, attention_mask = attention_mask)

            # assert output.dtype == torch.float32
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
            # print(torch.max(output), torch.min(output))
            # print(torch.max(target), torch.min(target))
            # loss = loss_fn(output.view(-1, 10), target.view(-1, 10))
            
            # #permute the output and target to match the shape of loss function
            output = output.permute(0, 2, 1)
            target = target.permute(0, 2, 1)

            #compute loss
            loss = loss_fn(output, target)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_data):.4f}")
        epoch_loss.append(total_loss / len(train_data))
    #plot loss
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.plot(epoch_loss)
    axs.set_title('Loss')
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Loss')
    #save plot 
    plt.savefig('loss.png')
    return model

import torch
import torch.nn as nn
import torch.optim as optim
import math 
class SudokuTransformerED(nn.Module):
    def __init__(self, input_size = 81, hidden_size = 10, output_size = 10, num_layers = 1, num_heads = 2, dropout = 0.1):
        super(SudokuTransformerED, self).__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, dropout=dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input , attention_mask = None):
        # input: [batch_size, seq_len]
        x = F.one_hot(input.long(), num_classes=10).float() # x = self.embedding(input)  # [batch_size, seq_len, hidden_size]
        x = self.pos_encoding(x)   # [batch_size, seq_len, hidden_size]

        print(x.shape)
        # encoder
        # encoder_out = self.transformer_encoder(x.permute(1, 0, 2))  # [seq_len, batch_size, hidden_size]
        encoder_out = x.permute(1, 0, 2)
        print("Vishal", encoder_out.shape)
        # decoder
        decoder_in = torch.zeros_like(x[:, 0, :]).unsqueeze(1)  # [batch_size, 1, hidden_size]
        print(decoder_in.shape)
        print("DONE")
        for i in range(input.shape[1]):
            decoder_out = self.transformer_decoder(decoder_in.permute(1, 0, 2), encoder_out)  # [1, batch_size, hidden_size]
            print(decoder_in.shape, decoder_out.shape)
            decoder_in = torch.cat([decoder_in, decoder_out], dim=1)  # [batch_size, i+2, hidden_size]

        out = self.linear(decoder_in)  # [batch_size, seq_len, output_size]
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)




class BiDirectionalSudokuTransformer(nn.Module):
    def __init__(self, d_model=10, num_layers=8, num_heads=10, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Positional encoding layer
        self.pos_enc = nn.Parameter(self.get_positional_encoding(81, d_model), requires_grad=False)

        # Input embedding layer
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=d_model)

        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList()
        for i in range(num_layers):
            # Apply two TransformerEncoderLayer modules, one in the forward direction and one in the reverse direction
            forward_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True, norm_first=False, dim_feedforward=d_model*4, dropout=dropout)
            backward_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True, norm_first=False, dim_feedforward=d_model*4, dropout=dropout)
            self.transformer_layers.append(nn.Sequential(forward_layer, backward_layer))

        # Output linear layer
        self.out = nn.Linear(d_model, 10)

    def forward(self, x, attention_mask=None):
        # Input shape: (batch_size, 81)

        #get one hot encoding of input
        x_one_hot = F.one_hot(x.long(), num_classes=10).float()  # Shape: (batch_size, 81, 10)

        x = x_one_hot + self.pos_enc[:x.shape[1], :].unsqueeze(0)

        assert (x.dtype == torch.float32)
        for i in range(self.num_layers):
            # Run the input sequence through the forward layer
            forward_output = self.transformer_layers[i][0](x)

            # Reverse the input sequence and run it through the backward layer
            backward_output = self.transformer_layers[i][1](torch.flip(x, [1]))

            # Reverse the output of the backward layer and concatenate with the output of the forward layer
            backward_output = torch.flip(backward_output, [1])
            #x = torch.cat((forward_output, backward_output), dim=-1)
            # sum of forward and backward
            x = forward_output + backward_output
            # print("Hello World")
        x = self.out(x)
        return x

    def get_positional_encoding(self, length, d_model):
        # Compute positional encoding as described in the paper
        position = torch.arange(length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pos_enc = torch.zeros((length, d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc

