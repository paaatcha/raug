import torch.nn as nn
from metablock import MetaBlock

class MetaBlockWithAttention(nn.Module):
    """
        Uso dos layers de atenção de para processamento extra das features dos dados de saída do MetaBlock
    """
    def __init__(self, feature_dim, meta_dim, nhead=2, num_transformer_layers=4):
        super(MetaBlockWithAttention, self).__init__()
        
        # Processar os dados com o bloco Metablock
        self.meta_block = MetaBlock(feature_dim, meta_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
            
    def forward(self, V, U):
        # Apply the MetaBlock; assume V shape: (batch, seq_len, feature_dim) and U shape: (batch, meta_dim)
        feat_metablock = self.meta_block(V, U)      
        # Acertar o formato do shape
        feat_metablock = feat_metablock.permute(0, 2, 1)
        # # Process with the Transformer Encoder layers
        feat_transformer = self.transformer_encoder(feat_metablock)
        
        # Permutar novamente, para manter o mesmo shape que o MetaBlock
        feat_transformer = feat_transformer.permute(0, 1, 2)
        
        return feat_metablock
