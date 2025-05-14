import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=bias)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            ConvLSTMCell(input_dim if i == 0 else hidden_dim,
                         hidden_dim,
                         kernel_size) for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input: (B, T, C, H, W)
        b, t, c, h, w = input.size()
        h_t, c_t = [], []
        for i in range(self.num_layers):
            h_t.append(torch.zeros(b, self.hidden_dim, h, w, device=input.device))
            c_t.append(torch.zeros(b, self.hidden_dim, h, w, device=input.device))

        outputs = []
        for time_step in range(t):
            x = input[:, time_step]
            for i, layer in enumerate(self.layers):
                h_t[i], c_t[i] = layer(x, h_t[i], c_t[i])
                x = self.dropout(h_t[i])
            outputs.append(h_t[-1])

        outputs = torch.stack(outputs, dim=1)  # (B, T, C, H, W)
        return outputs


class DeepEYEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 128, kernel_size=(11, 11, 1), stride=(4, 4, 1), padding=0),
            nn.Tanh(),
            nn.Conv3d(128, 64, kernel_size=(5, 5, 1), stride=(2, 2, 1), padding=0),
            nn.Tanh(),
        )

        self.convlstm = ConvLSTM(
            input_dim=64,
            hidden_dim=64,
            kernel_size=3,
            num_layers=1,
            dropout=0.4
        )

        self.convlstm2 = ConvLSTM(
            input_dim=64,
            hidden_dim=32,
            kernel_size=3,
            num_layers=1,
            dropout=0.3
        )

        self.convlstm3 = ConvLSTM(
            input_dim=32,
            hidden_dim=64,
            kernel_size=3,
            num_layers=1,
            dropout=0.5
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 128, kernel_size=(5, 5, 1), stride=(2, 2, 1)),
            nn.Tanh(),
            nn.ConvTranspose3d(128, 1, kernel_size=(11, 11, 1), stride=(4, 4, 1)),
            nn.Tanh(),
        )

    def forward(self, x):
        # x: (B, 1, 227, 227, 10)
        x = self.encoder(x)  # shape will be (B, 64, H, W, 10)
        x = x.permute(0, 4, 1, 2, 3)  # (B, T, C, H, W)
        x = self.convlstm(x)
        x = self.convlstm2(x)
        x = self.convlstm3(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, C, H, W, T)
        x = self.decoder(x)
        return x
