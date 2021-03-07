import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from rlkit.torch.core import PyTorchModule

################################################################################
################################################################################
"""                                 model                                    """
################################################################################
################################################################################
'''
class BatchTransitionAttention(PyTorchModule):
    def __init__(self, 
            vocab_size=1000,
            hidden=100,
            input_size=5,
            output_size=2,
            n_layers=3,
            attn_heads=1,
            dropout=0.1,
            use_multihead_attention=True,
            use_channel_attention=False,
            batch_size=1024
        ):

        self.save_init_params(locals())
        super().__init__()
        self.hidden = hidden
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.use_multihead_attention = use_multihead_attention
        self.batch_size=batch_size

        # multi-layers transformer blocks, deep network
        self.front=nn.Linear(self.input_size, self.hidden)
        if use_channel_attention:
            self.attention_blocks = nn.ModuleList(
                [CABlock(batch_size, hidden, hidden * 4, dropout) for _ in range(n_layers)])
        else:
            self.attention_blocks = nn.ModuleList(
                [TransformerBlock(hidden, attn_heads, hidden * 4, dropout, self.use_multihead_attention) for _ in range(n_layers)])
        
        #self.transformer_blocks = nn.ModuleList(
        #    [TransformerBlock(hidden, attn_heads, hidden * 4, dropout, self.use_multihead_attention) for _ in range(n_layers)])
        
        self.tail=nn.Linear(self.hidden, self.output_size)

    def forward(self, x, tanh=True):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        #x = x.unsqueeze(1)
        # running over multiple transformer blocks
        x = self.front(x)
        for transformer in self.attention_blocks:
            x = transformer.forward(x)
        if tanh:
            x = torch.tanh(self.tail(x))
        else:
            x = self.tail(x)
        #return x.squeeze(1)
        return x
'''
#class BERT(nn.Module):
class BERT(PyTorchModule):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """
    def __init__(self, 
            hidden=100,
            input_size=5,
            output_size=2,
            n_layers=6,
            attn_heads=128,
            dropout=0.1,
            use_sequence_attention=True,
            use_multihead_attention=True,
            use_channel_attention=True,
            mode='parallel',
            #batch_size=1024,
            #batch_attention=False
        ):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """
        self.save_init_params(locals())
        super().__init__()
        self.hidden = hidden
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.use_sequence_attention = use_sequence_attention
        self.use_multihead_attention = use_multihead_attention
        self.use_channel_attention = use_channel_attention
        self.mode = mode
        #self.batch_attention = batch_attention
        '''
        self.transition_attention = BatchTransitionAttention(
            vocab_size=vocab_size,
            hidden=128,
            input_size=input_size,
            output_size=input_size,
            n_layers=3,
            attn_heads=1,
            dropout=0.1,
            use_multihead_attention=False,
            use_channel_attention=True,
            batch_size=1024
        ) if self.batch_attention else None
        '''
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        #self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        #self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        m_front = [
            #nn.Linear(self.input_size, 16),
            nn.Linear(self.input_size, self.hidden),
            nn.Tanh(),
        ]
        self.front = nn.Sequential(*m_front)
        '''
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout, self.use_multihead_attention) for _ in range(n_layers)])
        '''
        '''
        if use_channel_attention:
            self.attention_blocks = nn.ModuleList(
                [CABlock(batch_size, hidden, hidden * 4, dropout) for _ in range(n_layers)])
        else:
            self.attention_blocks = nn.ModuleList(
                [TransformerBlock(hidden, attn_heads, hidden * 4, dropout, self.use_multihead_attention) for _ in range(n_layers)])
        '''
        if self.mode=='gate':
            #self.gate = ChannelAttention(feature_size=hidden, reduction=16)
            self.gate = BatchGate(feature_size=hidden, reduction=16)
            #self.gate = nn.ModuleList([
            #    ChannelAttention(feature_size=hidden, reduction=16, activation=nn.ReLU, output_activation=nn.ReLU),
            #    ChannelAttention(feature_size=hidden, reduction=16, activation=nn.ReLU, output_activation=nn.ReLU),
            #    ChannelAttention(feature_size=hidden, reduction=16, activation=nn.ReLU, output_activation=nn.Sigmoid)
            #])
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(
                hidden,
                attn_heads,
                hidden * 4,
                dropout,
                self.use_sequence_attention,
                self.use_multihead_attention,
                self.use_channel_attention,
                self.mode,
                ) for _ in range(n_layers)
            ])

        self.tail=nn.Linear(self.hidden, self.output_size)
        

    def forward(self, x, segment_info=None, tanh=True):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        '''
        if x.dim()==3:
            t, b, dim = x.size()
            x = x.view(t*b, 1, dim)
        elif x.dim()==2:
            x = x.unsqueeze(1)
        '''
        #t, b, dim = x.size()
        #if self.transition_attention:
        #    x = self.transition_attention(x)
        #x = x.view(t*b, 1, dim)

        #x = x.unsqueeze(-1)
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        
        #x = self.embedding(x, segment_info)
        #x = self.embedding(x)

        # running over multiple transformer blocks
        x = self.front(x)
        if self.mode=='gate':
            w = self.gate.forward(x, use_softmax=True, weight_only=True)
        #x = self.front(x).transpose(-2, -1)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x)
        if self.mode=='gate':
            x = x * w
        if tanh:
            x = torch.tanh(self.tail(x))
        else:
            x = self.tail(x)
        #x = torch.clamp(x, -1, 1)
        return x
'''
class ConvBERT(PyTorchModule):
    def __init__(self, 
            vocab_size=1000,
            hidden=100,
            input_size=5,
            output_size=2,
            n_layers=6,
            attn_heads=100,
            dropout=0.1,
            use_multihead_attention=True,
            use_channel_attention=False,
            batch_attention=False
        ):
        self.save_init_params(locals())
        super().__init__()
        self.hidden = hidden
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.use_multihead_attention = use_multihead_attention
        self.batch_attention = batch_attention

        
        #self.transition_attention = BatchTransitionAttention(
        #    vocab_size=vocab_size,
        #    hidden=100,
        #    input_size=input_size,
        #    output_size=input_size,
        #    n_layers=3,
        #    attn_heads=1,
        #    dropout=0.1
        #) if self.batch_attention else None
        

        # multi-layers transformer blocks, deep network
        m_front = [
            #nn.Linear(self.input_size, 16),
            nn.Conv1d(1, hidden, 1, padding=0, bias=True),
            nn.Tanh(),
        ]
        self.front = nn.Sequential(*m_front)
        if use_channel_attention:
            self.attention_blocks = nn.ModuleList(
                [CABlock(batch_size, hidden, hidden * 4, dropout) for _ in range(n_layers)])
        else:
            self.attention_blocks = nn.ModuleList(
                [TransformerBlock(hidden, attn_heads, hidden * 4, dropout, self.use_multihead_attention) for _ in range(n_layers)])
        
        #self.transformer_blocks = nn.ModuleList(
        #    [TransformerBlock(hidden, attn_heads, hidden * 4, dropout, self.use_multihead_attention) for _ in range(n_layers)])
        
        #self.tail=nn.Linear(self.hidden, self.output_size)
        m_tail=[
            nn.Conv1d(hidden, 1, 1, padding=0, bias=True),
            #nn.Tanh()
        ]
        self.tail = nn.Sequential(*m_tail)
        #self.tail2 = nn.Linear(16, self.output_size)

    def forward(self, x, segment_info=None, tanh=True):
        t, b, dim = x.size()
        
        #if self.transition_attention:
        #    x = self.transition_attention(x)
        
        x = x.view(t*b, 1, dim)
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        
        #x = self.embedding(x, segment_info)
        #x = self.embedding(x)

        # running over multiple transformer blocks
        x = self.front(x).transpose(-2, -1)
        for transformer in self.attention_blocks:
            x = transformer.forward(x)
        #x = self.tail1(x.transpose(-2, -1))
        x = x.transpose(-2, -1)
        if tanh:
            x = torch.tanh(self.tail(x))
        else:
            x = self.tail(x)
        #x = torch.clamp(x, -1, 1)
        return x
'''
class FlattenBERT(BERT):
    #if there are multiple inputs, concatenate along dim -1
    def forward(self, meta_size=16, batch_size=256, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1).unsqueeze(1).view(meta_size, batch_size, -1)
        #return super().forward(flat_inputs, **kwargs).squeeze(1)
        return super().forward(flat_inputs, **kwargs).view(meta_size*batch_size, -1)

################################################################################
################################################################################
"""                               embedding                                  """
################################################################################
################################################################################

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        #self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label=None):
        #x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]



################################################################################
################################################################################
"""                            TransformerBlock                              """
################################################################################
################################################################################
'''
class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, use_multihead_attention=True):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        if use_multihead_attention:
            self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        else:
            self.attention = Attention()
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
'''
class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(
        self,
        hidden,
        attn_heads,
        feed_forward_hidden,
        dropout,
        use_sequence_attention=True,
        use_multihead_attention=True,
        use_channel_attention=False,
        mode='parallel'
        ):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """
        super().__init__()
        self.use_channel_attention = use_channel_attention
        self.use_sequence_attention = use_sequence_attention
        self.mode = mode
        '''
        if use_multihead_attention:
            self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        else:
            self.attention = Attention()
        '''
        if self.use_sequence_attention:
            self.sequence_attention = MultiHeadedAttention(h=attn_heads, d_model=hidden) if use_multihead_attention else Attention()
        if self.use_channel_attention:
            self.batch_attention = ChannelAttention(feature_size=hidden, reduction=16)
        
        #if use_channel_attention:
        #    self.channel_attention_weight = ChannelAttentionWeight(channel=1024, feature_size=hidden, reduction=16, choose_conv=False)
        #    self.channel_input_sublayer = SublayerConnection(size=hidden, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        if self.mode=='serialize':
            self.extra_input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
    
    def tb_1_dim(self, x):
        t, b, dim = x.size()
        return x.view(t*b, 1, dim)

    def forward(self, x, mask=None):
        t, b, dim = x.size()
        if self.use_channel_attention and self.use_sequence_attention:
            if self.mode == 'parallel':
                x = self.input_sublayer(x, lambda _x: self.sequence_attention.forward(self.tb_1_dim(_x), self.tb_1_dim(_x), self.tb_1_dim(_x), mask=mask).view(t, b, dim)
                    + self.batch_attention.forward(_x) )
            elif self.mode == 'serialize':
                x = self.input_sublayer(x, lambda _x: self.sequence_attention.forward(self.tb_1_dim(_x), self.tb_1_dim(_x), self.tb_1_dim(_x), mask=mask).view(t, b, dim))
                x = self.extra_input_sublayer(x, lambda _x: self.batch_attention.forward(_x))
        elif self.use_channel_attention:
            x = self.input_sublayer(x, lambda _x: self.batch_attention.forward(_x))
        elif self.use_sequence_attention:
            x = self.input_sublayer(x, lambda _x: self.sequence_attention.forward(self.tb_1_dim(_x), self.tb_1_dim(_x), self.tb_1_dim(_x), mask=mask).view(t, b, dim))

        x = self.output_sublayer(x, self.feed_forward)

        return self.dropout(x)

class CABlock(nn.Module):

    def __init__(self, hidden, feed_forward_hidden, dropout=0.1, reduction=16):
        super().__init__()
        self.attention = ChannelAttention(feature_size=hidden, reduction=16)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()
        #self.activation = nn.LeakyReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, residual=True):
        "Apply residual connection to any sublayer with the same size."
        if residual:
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            return self.dropout(sublayer(self.norm(x)))
        #return x + self.dropout(sublayer(x))

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

################################################################################
################################################################################
"""                               attention                                  """
################################################################################
################################################################################
class BatchGate(nn.Module):
    """
    Compute Batch Attention using channel attention
    """
    #def __init__(self, feature_size=128, reduction=16, choose_conv=False):
    def __init__(self, feature_size=128, reduction=16, activation=nn.Tanh, output_activation=nn.Sigmoid):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # feature channel downscale and upscale --> channel weight
        self.fc = nn.Sequential(
            nn.Linear(feature_size, feature_size//reduction),
            activation(),
            nn.Linear(feature_size//reduction, feature_size),
            activation(),
            #LayerNorm(feature_size),

            nn.Linear(feature_size, feature_size//reduction),
            activation(),
            nn.Linear(feature_size//reduction, feature_size),
            activation(),
            #LayerNorm(feature_size),

            nn.Linear(feature_size, feature_size//reduction),
            activation(),
            nn.Linear(feature_size//reduction, feature_size),
            #nn.Sigmoid()
            output_activation()
            #nn.Hardtanh()
        )
    def forward(self, x, use_softmax=False, weight_only=False):
        y = self.fc(x)
        y = self.avg_pool(y)
        if use_softmax:
            y = F.softmax(y, dim=1)
        '''
        y = self.avg_pool(x)
        if self.choose_conv:
            y = self.conv_du(y)
        else:
            y = y.squeeze(-1)
            y = self.fc_du(y)
            y = y.unsqueeze(-1)
        '''
        if weight_only:
            return y
        return x * y

class ChannelAttention(nn.Module):
    """
    Compute Batch Attention using channel attention
    """
    #def __init__(self, feature_size=128, reduction=16, choose_conv=False):
    def __init__(self, feature_size=128, reduction=16, activation=nn.Tanh, output_activation=nn.Sigmoid):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # feature channel downscale and upscale --> channel weight
        '''
        self.choose_conv = choose_conv
        if self.choose_conv:
            self.conv_du = nn.Sequential(
                nn.Conv1d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.Tanh(),
                nn.Conv1d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
            )
        else:
            self.fc_du = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.Tanh(),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
            )
        '''
        self.fc = nn.Sequential(
            nn.Linear(feature_size, feature_size//reduction),
            #nn.Tanh(),
            #nn.Hardtanh(),
            activation(),
            #nn.Linear(feature_size//reduction, feature_size//reduction),
            #nn.Tanh(),
            #activation(),
            nn.Linear(feature_size//reduction, feature_size),
            #nn.Sigmoid()
            output_activation()
            #nn.Hardtanh()
        )
    def forward(self, x, use_softmax=False, weight_only=False):
        y = self.fc(x)
        y = self.avg_pool(y)
        if use_softmax:
            y = F.softmax(y, dim=1)
        '''
        y = self.avg_pool(x)
        if self.choose_conv:
            y = self.conv_du(y)
        else:
            y = y.squeeze(-1)
            y = self.fc_du(y)
            y = y.unsqueeze(-1)
        '''
        if weight_only:
            return y
        return x * y

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):

        #print(query.shape, key.shape, value.shape)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        
        #print("score", scores.shape)
    

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value)


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
