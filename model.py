import torch
from torch import nn


class D1Model(nn.Module):
    def __init__(self, config):
        super(D1Model, self).__init__()

        self.config = config

        self.layer1_1 = nn.LSTM(input_size=config.sequence_length, hidden_size=config.hidden_dim_1d, num_layers=1, bidirectional=True)
        self.layer1_2 = nn.LSTM(input_size=config.sequence_length, hidden_size=config.hidden_dim_1d, num_layers=1, bidirectional=True)

        self.layer2_1 = nn.Linear(config.hidden_dim_1d * 2, config.hidden_dim_1d)
        self.layer2_2 = nn.Linear(config.hidden_dim_1d * 2, config.hidden_dim_1d)

        self.layer3_1 = nn.Linear(config.hidden_dim_1d, config.mid_dim)
        self.layer3_2 = nn.Linear(config.hidden_dim_1d, config.mid_dim)

        self.layer4 = nn.Linear(config.mid_dim * 2, config.out_dim)

        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):

        self.layer1_1.reset_parameters()
        self.layer1_2.reset_parameters()

        nn.init.xavier_uniform_(self.layer2_1.weight)
        self.layer2_1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.layer2_2.weight)
        self.layer2_2.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.layer3_1.weight)
        self.layer3_1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.layer3_2.weight)
        self.layer3_2.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.layer4.weight)
        self.layer4.bias.data.fill_(0)

    def forward(self, batch_data):
        """
        :param: batch_data_1d
        :return: outputs, logits, hidden_states
        """
        ids1 = batch_data.ids1
        ids2 = batch_data.ids2

        # Parameter-sharing
        if self.config.shared == True:
            h1 = torch.zeros(2, self.config.batch_size, self.config.hidden_dim_1d).cuda()
            c1 = torch.zeros(2, self.config.batch_size, self.config.hidden_dim_1d).cuda()
            input1 = ids1.view(self.config.batch_size, self.config.sequence_length).unsqueeze(0)
            # ids shape: [sequence_length, batch size, hidden_dim]
            output1, (_, _) = self.layer1_1(input1.float(), (h1, c1))

            h2 = torch.zeros(2, self.config.batch_size, self.config.hidden_dim_1d).cuda()
            c2 = torch.zeros(2, self.config.batch_size, self.config.hidden_dim_1d).cuda()
            input2 = ids2.view(self.config.batch_size, self.config.sequence_length).unsqueeze(0)
            output2, (_, _) = self.layer1_1(input2.float(), (h2, c2))

            # output1 shape (8, 256)
            self.output1 = torch.squeeze(output1, dim=0)
            # output2 shape (8, 256)
            self.output2 = torch.squeeze(output2, dim=0)

            # logits1 shape (8, 64)
            self.logits1 = self.dropout(self.relu(self.layer2_1(self.output1)))
            # logits2 shape (8, 64)
            self.logits2 = self.dropout(self.relu(self.layer2_1(self.output2)))

            # logits1 shape (8, 4)
            self.logits1 = self.dropout(self.relu(self.layer3_1(self.logits1)))
            # logits2 shape (8, 4)
            self.logits2 = self.dropout(self.relu(self.layer3_1(self.logits2)))


        # Parameter-independent
        else:
            h1 = torch.zeros(2, self.config.batch_size, self.config.hidden_dim_1d).cuda()
            c1 = torch.zeros(2, self.config.batch_size, self.config.hidden_dim_1d).cuda()
            input1 = ids1.view(self.config.batch_size, self.config.sequence_length).unsqueeze(0)
            output1, (_, _) = self.layer1_1(input1.float(), (h1, c1))

            h2 = torch.zeros(2, self.config.batch_size, self.config.hidden_dim_1d).cuda()
            c2 = torch.zeros(2, self.config.batch_size, self.config.hidden_dim_1d).cuda()
            input2 = ids2.view(self.config.batch_size, self.config.sequence_length).unsqueeze(0)
            output2, (_, _) = self.layer1_2(input2.float(), (h2, c2))

            # output1 shape (8, 256)
            self.output1 = torch.squeeze(output1, dim=0)
            # output2 shape (8, 256)
            self.output2 = torch.squeeze(output2, dim=0)

            # logits1 shape (8, 64)
            self.logits1 = self.dropout(self.relu(self.layer2_1(self.output1)))
            # logits2 shape (8, 64)
            self.logits2 = self.dropout(self.relu(self.layer2_2(self.output2)))

            # logits1 shape (8, 4)
            self.logits1 = self.dropout(self.relu(self.layer3_1(self.logits1)))
            # logits2 shape (8, 4)
            self.logits2 = self.dropout(self.relu(self.layer3_2(self.logits2)))

        self.hidden_states = torch.cat((self.logits1, self.logits2), dim=1) # 1D hidden_states shape (8, 8)

        self.logits = self.layer4(self.hidden_states) # logits shape (8, 1)
        self.output = self.sigmoid(self.logits)

        return self.output, self.hidden_states



