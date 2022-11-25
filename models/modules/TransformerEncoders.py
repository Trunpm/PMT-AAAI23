import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """
    def __init__(self, d_model, dropout=0.1, max_len=128):
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
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PositionalEncodingLearned1D(nn.Module):
    """
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncodingLearned1D(d_model)
    """
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(PositionalEncodingLearned1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pos_embed = nn.Embedding(max_len, d_model)
        nn.init.uniform_(self.pos_embed.weight)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        idx = torch.arange(x.shape[0],device=x.device)
        x = x + self.pos_embed(idx).unsqueeze(1)
        return self.dropout(x)


        


class TransformerEncoderLayer_QKV(nn.Module):
    """
    Completed TransformerEncoderLayer for input Q K V

    """
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, res_dropout=0.1, activ_dropout=0.1):
        super(TransformerEncoderLayer_QKV,self).__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads,dropout=attn_dropout)
        self.res_dropout = res_dropout
        self.activ_dropout = activ_dropout
        self.normalize_before = True

        self.fc1 = nn.Linear(embed_dim, 4*embed_dim)
        self.fc2 = nn.Linear(4*embed_dim, embed_dim)
        self.activ=nn.ELU()
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(2)])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_key_padding_mask(self, max_length, key_len):
        #return shape (batch, max_length)
        return torch.arange(0,max_length,device=key_len.device).unsqueeze(0).expand(key_len.shape[0],max_length).ge(key_len.unsqueeze(1)).bool()

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x
  
    def forward(self, x_q, x_k=None, x_v=None, kv_len=None):
        """
        Args:
            x_q    (Tensor):         input to the layer of shape            (seq_len,  batch, embed_dim)
            x_k    (None or Tensor): if Tensor, input to the layer of shape (seq_len', batch, embed_dim)
            x_v    (None or Tensor): if Tensor, input to the layer of shape (seq_len', batch, embed_dim)
            kv_len (None or Tensor): if Tensor, input to the layer of shape (batch,)
        Returns:
            out shape (seq_len,  batch, embed_dim)
        """
        residual = x_q
        x_q = self.maybe_layer_norm(0, x_q, before=True)
        if x_k is None:
            key_padding_mask = self._get_key_padding_mask(x_q.shape[0],kv_len) if kv_len is not None else None
            x_q, _ = self.self_attn(query=x_q, key=x_q, value=x_q, key_padding_mask=key_padding_mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            key_padding_mask = self._get_key_padding_mask(x_k.shape[0],kv_len) if kv_len is not None else None
            if x_v is None:
                x_q, _ = self.self_attn(query=x_q, key=x_k, value=x_k, key_padding_mask=key_padding_mask)
            else:
                x_v = self.maybe_layer_norm(0, x_v, before=True)
                x_q, _ = self.self_attn(query=x_q, key=x_k, value=x_v, key_padding_mask=key_padding_mask)
             
        x_q = F.dropout(x_q, p=self.res_dropout, training=self.training)
        x_q = residual + x_q
        x_q = self.maybe_layer_norm(0, x_q, after=True)

        residual = x_q
        x_q = self.maybe_layer_norm(1, x_q, before=True)
        x_q = self.activ(self.fc1(x_q))
        x_q = F.dropout(x_q, p=self.activ_dropout, training=self.training)
        x_q = self.fc2(x_q)
        x_q = F.dropout(x_q, p=self.res_dropout, training=self.training)
        x_q = residual + x_q
        x_q = self.maybe_layer_norm(1, x_q, after=True)
        return x_q




class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, res_dropout=0.1, activ_dropout=0.1, num_layers=1):
        super().__init__()

        self.layers = nn.ModuleList([])
        for layer in range(num_layers):
            new_layer = TransformerEncoderLayer_QKV(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                res_dropout=res_dropout,
                                                activ_dropout=activ_dropout)
            self.layers.append(new_layer)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_q, x_k=None, x_v=None, kv_len=None):
        """
        Args:
            x_q    (Tensor):         input to the layer of shape            (seq_len,  batch, embed_dim)
            x_k    (None or Tensor): if Tensor, input to the layer of shape (seq_len', batch, embed_dim)
            x_v    (None or Tensor): if Tensor, input to the layer of shape (seq_len', batch, embed_dim)
            kv_len (None or Tensor): if Tensor, input to the layer of shape (batch,)
        Returns:
            out shape (seq_len, batch, embed_dim)
        """
         # encoder layers
        intermediates = [x_q]
        for layer in self.layers:
            x_q = layer(x_q, x_k, x_v, kv_len)
            intermediates.append(x_q)

        x_q = self.layer_norm(x_q)
        return x_q














class Block2(nn.Module):
    def __init__(self, embed_dim, res_dropout=0.1):
        super(Block2,self).__init__()
        self.scaling = embed_dim ** -0.5
        self.res_dropout = res_dropout
        self.activ=nn.ELU()

        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(3)])
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc_ = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fcout = nn.Linear(embed_dim, embed_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_key_padding_mask(self, max_length, key_len):
        #return shape (batch, max_length)
        return torch.arange(0,max_length,device=key_len.device).unsqueeze(0).expand(key_len.shape[0],max_length).ge(key_len.unsqueeze(1)).long()

    def forward(self, x_v1, x_, x_v2):
        """
        Args:
            x_v1    (Tensor): input to the layer of shape            (seq_len1, batch, embed_dim)
            x_      (Tensor): if Tensor, input to the layer of shape (seq_len', batch, embed_dim)
            x_v2    (Tensor): if Tensor, input to the layer of shape (seq_len2, batch, embed_dim)
        Returns:
            out shape (seq_len1, batch, embed_dim)
        """
        residual = x_v1
        x_v1 = self.layer_norms[0](x_v1)
        x_ = self.layer_norms[1](x_)
        x_v2 = self.layer_norms[2](x_v2)

        attn_weights_to_ = torch.bmm(self.activ(self.fc_(x_)).transpose(0,1), self.activ(self.fc2(x_v2)).permute(1,2,0))*self.scaling
        attn_weights_to_ = F.softmax(attn_weights_to_, dim=-1) #B seq_len' seq_len2
        attn_weights_tov1 = torch.bmm(self.activ(self.fc1(x_v1)).transpose(0,1), self.activ(self.fc_(x_)).permute(1,2,0))*self.scaling
        attn_weights_tov1 = F.softmax(attn_weights_tov1, dim=-1) #B seq_len1 seq_len'
        x_v1 = torch.bmm(attn_weights_tov1,torch.bmm(attn_weights_to_,x_v2.transpose(0,1))).transpose(0,1)

        x_v1 = self.activ(self.fcout(x_v1))
        x_v1 = F.dropout(x_v1, p=self.res_dropout, training=self.training)
        x_v1 = residual + x_v1
        return x_v1