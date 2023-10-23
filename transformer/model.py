import torch 
import math
class InputEmbeddings(torch.nn.Module):
    def __init__(self,d_model: int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size 
        self.embedding = torch.nn.Embedding(vocab_size,d_model) 

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(torch.nn.Module):
    def __init__(self,d_model:int, seq_len:int ,dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = torch.nn.Dropout(dropout)

        #Create a matrix of shape seqlen,d_model 
        pe = torch.zeros(seq_len,d_model)
        #Create a vector of shape (seqlen,1)
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()* (-math.log(10000.0)/d_model))
        #Apply sin to even pos and cos to odd 
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe.unsqueeze(0) #(1,seqlen,d_model)
        self.register_buffer('pe',pe)


    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)

#Add and Norm
#Take a batch, each item has features, calc mean and var for each item
class LayerNormalization(torch.nn.Module):
    def __init__(self,eps: float = 10**-6 ) -> None:
        super().__init__()
        self.eps = eps 
        #Alpha is multiplicative, bias is additive
        self.alpha = torch.nn.Parameter(torch.ones(1))
        self.bias  = torch.nn.Parameter(torch.zeros(1)) 
    
    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        std =x.std(dim = -1, keepdim = True)
        return self.alpha *(x-mean)/math.sqrt(std+self.eps) + self.bias


class FeedForwardBlock(torch.nn.Module): 
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(d_model,d_ff) # W1 and B1 
        self.dropout = torch.nn.Dropout(dropout)
        self.linear_2 = torch.nn.Linear(d_ff, d_model) 

    
    def forward(self,x):
        # (Batch sqlen dmodel) -> batch seqlen dff -> batch seqlen dmodel 
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

            
#Multiheaded attention:
#QKV Query key values 
#
class MultiHeadAttentionBlock(torch.nn.Module):
    #We need dropout, h for no.of heads, and d_model 
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = dropout 
        #Ensure that d_model is divisible by h 
        assert d_model % h == 0, "d_model needs to be divisible by h"

        self.d_k = d_model // h

        self.w_q = torch.nn.Linear(d_model,d_model) #y = xA_t + b : seqlen, dmodel  @ dmodel , dmodel -> s
        self.w_k = torch.nn.Linear(d_model,d_model)
        self.w_v = torch.nn.Linear(d_model,d_model)

        self.w_o = torch.nn.Linear(d_model,d_model) 
        self.dropout = torch.nn.Dropout(dropout)

    @staticmethod
    def attention(query,key,value,mask,dropout:torch.nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_filled_(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1 )
        if dropout is not None: 
            attention_scores = dropout(attention_scores)
        return (attention_scores@value,attention_scores)

    def forward(self,q,k,v,mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        #Divide qkv into smaller matrices  

        query = query.view(query.shape[0],query.shape[1],self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h, self.d_k).transpose(1,2)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask, self.dropout)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)

        return self.w_o(x)

class ResidualConnection(torch.nn.Module):
    def __init__(self,dropout:float) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(torch.nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = torch.nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x





class Encoder(torch.nn.Module):
    def __init__(self, features: int, layers: torch.nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class DecoderBlock(torch.nn.Module):
    def __init__(self,self_attention_block: MultiHeadAttentionBlock, 
                 cross_attention_block:MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block 
        self.residual_connections = torch.nn.Module([ResidualConnection(dropout) for _ in range(3)])
    

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x,lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connections[2](x,self.feed_forward_block)
        return x



class Decoder(torch.nn.Module):
    def __init__(self, layers:torch.nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)

class ProjectionLayer(torch.nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x),dim=-1)
    

class Transformer(torch.nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self,src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self,encoder_output, src_mask,tgt,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    

    def project(self,x):
        return self.projection_layer(x)



#Build based on usecase
def build(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, torch.nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, torch.nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    
    return transformer