
class AttnDecoderRNN(nn.Module):
    def __init__(self, dec_hidden_size, enc_hidden_size, dec_output_size, dropout_p=0):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = dec_hidden_size
        self.output_size = dec_output_size
        self.dropout_p = dropout_p

        self.lin_decoder = nn.Linear(dec_hidden_size, dec_hidden_size) # to convert input
        self.lin_encoder = nn.Linear(enc_hidden_size, dec_hidden_size) # to convert encoder_outputs

        self.linear_out = nn.Linear(dec_hidden_size + enc_hidden_size , dec_hidden_size, bias=False)

    def forward(self, input, hidden, encoder_outputs):
        # hidden not a tuple (htx, ctx) -> batch X dec_hid
        # encoder_outputs -> batch X seq X enc_hid
        # input -> batch x 1 X dec_hid
        projected_context = F.tanh(self.lin_encoder(encoder_outputs)) # # batch X seq X enc_hid -> batch X seq X dec_hid
        projected_input = F.tanh(self.lin_decoder(input)).unsqueeze(2)  # batch X dec_hid X 1
        # RAW ATTENTION
        attn_dot = torch.bmm(projected_context, projected_input).squeeze(2) # batch X seq X 1 -> batch X seq
        attn = F.softmax(attn_dot, dim=1) # NORMALIZED ATTENTION
        reshaped_attn = attn.unsqueeze(1) # batch X 1 X seq
        weighted_context = torch.bmm(reshaped_attn, encoder_outputs).squeeze(1)  # batch X 1 X seq * batch X seq X enc_hid
                                                                        # -> batch X 1 x enc_hid -> batch X enc_hid

        h_tilde = torch.cat((weighted_context, hidden), 1)  #  -> batch X dec_hid+enc_hid
        h_tilde = F.tanh(self.linear_out(h_tilde))

        return h_tilde, attn
