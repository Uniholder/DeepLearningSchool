import random
import torch
from torch import nn
from torch.nn import functional as F


def softmax(x, temperature=10): # use your temperature
    e_x = torch.exp(x / temperature)
    return e_x / torch.sum(e_x, dim=0)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(  # LSTM
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        '''
        src: [src sent len, batch size]
        '''
        # Compute an embedding from the src data and apply dropout to it
        # embedded: [src sent len, batch size, emb dim]
        embedded = self.dropout(self.embedding(src))
        
        # Compute the RNN output values of the encoder RNN. 
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        # outputs: [src sent len, batch size, hid dim * n directions]
        # hidden: [n layers * n directions, batch size, hid dim]
        # cell: [n layers * n directions, batch size, hid dim]
        outputs, hidden = self.rnn(embedded)  # cell for LSTM

        # outputs are always from the top hidden layer
        if self.bidirectional:
            hidden = hidden.reshape(self.n_layers, 2, -1, self.hid_dim)
            hidden = hidden.transpose(1, 2).reshape(self.n_layers, -1, 2 * self.hid_dim)
            # cell = cell.reshape(self.n_layers, 2, -1, self.hid_dim)
            # cell = cell.transpose(1, 2).reshape(self.n_layers, -1, 2 * self.hid_dim)
        return outputs, hidden  # , cell


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, enc_hid_dim)
        self.v = nn.Linear(enc_hid_dim, 1)
        
    def forward(self, hidden, encoder_outputs):
        '''
        hidden: [1, batch size, dec_hid_dim]
        encoder_outputs: [src sent len, batch size, enc_hid_dim]
        '''
        # repeat hidden
        sent_len = encoder_outputs.shape[0]
        H = hidden.repeat(sent_len, 1, 1)
        # concatenate H with encoder_outputs and calculate energy
        E = torch.tanh(self.attn(torch.cat((H, encoder_outputs), dim=2)))
        # get attention, use softmax function which is defined, can change temperature
        a = self.v(E)
        
        return softmax(a)
    
    
class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
         # use GRU
        self.rnn = nn.GRU(  # LSTM
            input_size=emb_dim+enc_hid_dim,
            hidden_size=dec_hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )

        # linear layer to get next word
        self.out = nn.Linear(dec_hid_dim+emb_dim+enc_hid_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, input_, hidden, encoder_outputs):
        '''
        input_: [batch size]
        hidden: [n layers * n directions, batch size, hid dim]
        n directions in the decoder will both always be 1, therefore:
        hidden: [n layers, batch size, hid dim]
        encoder_outputs: [src sent len, batch size, enc_hid_dim * n directions]
        '''
        # input: [1, batch size]
        input_ = input_.unsqueeze(0) # because only one word, no words sequence
        
        # embedded: [1, batch size, emb dim]
        embedded = self.dropout(self.embedding(input_))
        
        # get weighted sum of encoder_outputs
        a = self.attention(hidden[0].squeeze(0), encoder_outputs)
        weighted = (encoder_outputs * a).sum(dim=0).unsqueeze(0)
        # concatenate weighted sum and embedded, break through the GRU
        output, hidden = self.rnn(torch.cat([embedded, weighted], dim=2), hidden)
        # get predictions
        # prediction: [batch size, output dim]
        prediction = self.out(torch.cat([output.squeeze(0), embedded.squeeze(0), weighted.squeeze(0)], dim=1))

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        if encoder.bidirectional:
            assert encoder.hid_dim * 2 == decoder.dec_hid_dim, \
                "Hidden dimensions of encoder and decoder must be equal!"
        else:
            assert encoder.hid_dim == decoder.dec_hid_dim, \
                "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        '''
        # src: [src sent len, batch size]
        # trg: [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        '''
        # Again, now batch is the first dimension instead of zero
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_states, hidden = self.encoder(src)  # cell for LSTM
        
        #first input to the decoder is the <sos> tokens
        input_ = trg[0, :]
        
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden = self.decoder(input_, hidden, enc_states)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(-1) 

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input_ = trg[t] if teacher_force else top1
        
        return outputs
