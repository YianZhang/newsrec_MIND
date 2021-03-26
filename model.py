import math
import torch
from torch import nn
from transformers import AutoModel
from transformers.models.bert.modeling_bert import BertSelfAttention

# program counter
# 查data对不对
# 看<5的多不多，不多直接删除？
# notes: scatter, gather
# pooling 的时候也是5个一起做
# gradient clipping 1
# AdamW, scheduler
# most recent history + position embedding

class MySelfAttention(BertSelfAttention):
  # init: (self, config)
  def convert_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask[:, None, None, :]
        return ((1-extended_attention_mask) * -10000.0).to(dtype=next(self.parameters()).dtype)

  def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + self.convert_attention_mask(attention_mask)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
        

class Attention_pooling(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        # input shape: (batch_size, his_size, emb_size)
        # self.projection = torch.nn.Linear(dim1, dim2) # (batch_size, his_size, dim2)
        # self.query = torch.nn.Linear(dim2, 1, bias = False) # (batch_size, his_size, 1)
        self.softmax = nn.Softmax(dim = -2)

        self.projection = torch.nn.Linear(dim, 1, bias = False)

    def forward(self, x):
        # input shape: (batch_size, his_size, emb_size)

        # weights = nn.Softmax(self.query(torch.tanh(self.projection(x))), dim=-1) # the softmax dimension problem
        weights = self.softmax(self.projection(x)) # batch_size, his_size

        return torch.sum(torch.mul(x, weights), dim=-2, keepdim=False) # batch, emb_size



class NewsRec(torch.nn.Module):
    def __init__(self, self_attention_config, ht_model='bert-base-uncased'):
        super().__init__()
        self.news_encoder = AutoModel.from_pretrained(ht_model)
        self.news_MHA = MySelfAttention(self_attention_config)
        self.news_pooling = Attention_pooling(self.news_encoder.config.hidden_size)

    def forward(self, x, batch_size):
        #print(x.keys())
        labels, candidate_mask, history_mask = x['labels'], x['candidate_mask'], x['history_mask']
        del x['labels']; del x['candidate_mask']; del x['history_mask']

        his_indices = torch.nonzero(labels == -1, as_tuple = True)[0]
        impr_indices = torch.nonzero(labels != -1, as_tuple = True)[0]
        
        news_reprs = self.news_encoder(**x).pooler_output
        his_reprs = news_reprs[his_indices].view(batch_size, -1, news_reprs.shape[-1])
        # impr_labels = labels[impr_indices].view(batch_size, -1)
        impr_reprs = news_reprs[impr_indices].view(batch_size, -1, news_reprs.shape[-1])

        self_attended_his_reprs = self.news_MHA(his_reprs)[0]

        user_reprs = self.news_pooling(self_attended_his_reprs)

        # print(user_reprs.shape, impr_reprs.shape) # checked. shapes are correct

        scores = torch.bmm(impr_reprs, user_reprs.view(user_reprs.shape[0], user_reprs.shape[1], 1))

        # print(user_reprs[1], impr_reprs[1][1], scores) # checked. bmm is correct.

        return scores


if __name__ == '__main__':
    # HPs
    # position embedding related HPs are useless.
    self_attention_hyperparameters = {'num_attention_heads' : 2, 'hidden_size' : 768, 'attention_probs_dropout_prob': 0.2, 'max_position_embeddings': 4, 'is_decoder': False, 'position_embedding_type' : None}
    BATCH_SIZE = 2

    # get data
    from data_loading import MINDDataset
    from torch.utils.data import RandomSampler
    from torch.utils.data import DataLoader
    from utils import Config
    train = MINDDataset('news.tsv', 'behaviors.tsv',batch_size=BATCH_SIZE)
    train.load_data()
    train_sampler = RandomSampler(my_ds)
    train_dataloader = DataLoader(
    my_ds,
    sampler=train_sampler,
    batch_size=my_ds.batch_size,
    collate_fn=my_ds.collate_fn
    )

    valid = MINDDataset('news.tsv', 'behaviors.tsv',batch_size=BATCH_SIZE)
    train.load_data()
    train_sampler = RandomSampler(my_ds)
    train_dataloader = DataLoader(
    my_ds,
    sampler=train_sampler,
    batch_size=my_ds.batch_size,
    collate_fn=my_ds.collate_fn
    )

    # build the model
    self_attention_config = Config(self_attention_hyperparameters)
    model = NewsRec(self_attention_config)

    # training





    # attn_params = {
    #     # 'embed_dim': 768,
    #     'num_heads' : 4, 
    #     'dropout' : 0.2, 
    #     'intermediate_dimension' : 200,
    # }

    batch = next(iter(train_dataloader))
    model(batch, BATCH_SIZE)
    # print(model(next(iter(train_dataloader)), BATCH_SIZE).shape)

