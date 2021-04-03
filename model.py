import math
import torch
from torch import nn
from transformers import AutoModel
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers import get_linear_schedule_with_warmup

# program counter
# 查data对不对
# 看<5的多不多，不多直接删除？
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
    
    def masking(self, x, attention_mask):
        mask = ((1 - attention_mask) * -10000.0).to(dtype=next(self.parameters()).dtype)
        return x + mask.view(mask.shape[0], mask.shape[1], 1)

    def forward(self, x, attention_mask):
        # input shape: (batch_size, his_size, emb_size)
        # attention_mask shape: batch_size, his_size

        # weights = nn.Softmax(self.query(torch.tanh(self.projection(x))), dim=-1) # the softmax dimension problem
        weights = self.softmax(self.masking(self.projection(x), attention_mask)) # batch_size, his_size, 1

        return torch.sum(torch.mul(x, weights), dim=-2, keepdim=False) # batch, emb_size



class NewsRec(torch.nn.Module):
    def __init__(self, self_attention_config, ht_model='bert-base-uncased'):
        super().__init__()
        self.news_encoder = AutoModel.from_pretrained(ht_model)
        self.news_MHA = MySelfAttention(self_attention_config)
        self.news_pooling = Attention_pooling(self.news_encoder.config.hidden_size)
    
    def masking(self, x, candidate_mask):
        mask = ((1 - candidate_mask) * -10000.0).to(dtype=next(self.parameters()).dtype)
        return x + mask

    def forward(self, x):
        #print(x.keys())
        batch_size = x['candidate_mask'].shape[0]
        labels, candidate_mask, history_mask = x['labels'], x['candidate_mask'], x['history_mask']
        del x['labels']; del x['candidate_mask']; del x['history_mask']

        his_indices = torch.nonzero(labels == -1, as_tuple = True)[0]
        impr_indices = torch.nonzero(labels != -1, as_tuple = True)[0]
        
        news_reprs = self.news_encoder(**x).pooler_output
        his_reprs = news_reprs[his_indices].view(batch_size, -1, news_reprs.shape[-1])
        # impr_labels = labels[impr_indices].view(batch_size, -1)
        impr_reprs = news_reprs[impr_indices].view(batch_size, -1, news_reprs.shape[-1])

        self_attended_his_reprs = self.news_MHA(his_reprs, history_mask)[0]

        user_reprs = self.news_pooling(self_attended_his_reprs, history_mask)

        # print(user_reprs.shape, impr_reprs.shape) # checked. shapes are correct

        scores = torch.bmm(impr_reprs, user_reprs.view(user_reprs.shape[0], user_reprs.shape[1], 1)).squeeze(-1)

        # print(user_reprs[1], impr_reprs[1][1], scores) # checked. bmm is correct.

        return self.masking(scores, candidate_mask)


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    # model HPs
    # position embedding related HPs are useless.
    BATCH_SIZE = 16
    self_attention_hyperparameters = {'num_attention_heads' : 16, 'hidden_size' : 768, 'attention_probs_dropout_prob': 0.2, 'max_position_embeddings': 4, 'is_decoder': False, 'position_embedding_type' : None}
    assert self_attention_hyperparameters['hidden_size'] % self_attention_hyperparameters['num_attention_heads'] == 0
    # get data
    from data_loading import MINDDataset
    from torch.utils.data import RandomSampler
    from torch.utils.data import DataLoader
    from utils import Config, save_checkpoint, load_checkpoint
    train = MINDDataset('train/news.tsv', 'train/behaviors.tsv',batch_size=BATCH_SIZE)
    train.load_data()
    train_sampler = RandomSampler(train)
    train_dataloader = DataLoader(
    train,
    sampler=train_sampler,
    batch_size=train.batch_size,
    collate_fn=train.collate_fn
    )

    valid = MINDDataset('valid/news.tsv', 'valid/behaviors.tsv',batch_size=BATCH_SIZE)
    valid.load_data()
    valid_sampler = RandomSampler(valid)
    valid_dataloader = DataLoader(
    valid,
    sampler=valid_sampler,
    batch_size=valid.batch_size,
    collate_fn=valid.collate_fn
    )

    # build the model
    self_attention_config = Config(self_attention_hyperparameters)
    model = NewsRec(self_attention_config).to(device)



    # training
    # BERT parameters:
    # batch size: 16, 32
    # lr(Adam): 5e-5, 3e-5, 2e-5
    # number of epochs: 2,3,4
    # dropout: 0.1

    MAX_EPOCHS = 5
    lr = 3e-5
    num_warmup_steps = 3000 # bert 10,000
    checkpointing_freq = 200
    valid_used_ratio = 0.005 # out of 6962 * 16 # change when swtiching to large datasets!

    num_train_steps = MAX_EPOCHS*(len(train_dataloader) - 1) # to be further checked
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
    criterion = torch.nn.CrossEntropyLoss()
    labels = torch.tensor([0] * BATCH_SIZE).to(device)
    
    



    try:
        load_checkpoint(model, optimizer, device)
        print('checkpoint loaded')
    except:
        print('failed to load any checkpoints.')

    #04/02: check nrms HPs; npratio

    lowest_loss = 100000
    for epoch in range(MAX_EPOCHS):
        #TODO: early stopping and checkpointing
        #TODO: shuffling
         # transformers AdamW (opt), scheduler (get linear schedule with warmup from transformers)
            # cross-entropy by batch?
            # clipping
            # shuffling will slow things down
            # early stopping using metrics
            # lr, batch_size, max_epochs, 
            # bert config HPs
        total_loss = 0
        model.train()
        for batch_id, data_batch in enumerate(train_dataloader):
            if batch_id == len(train_dataloader) - 1:
                continue
            # impr_indices = torch.nonzero(data_batch["labels"] != -1, as_tuple = True)[0] # must retrieve the labels first, because it is deleted by the forward func
            # impr_labels = data_batch["labels"][impr_indices].view(BATCH_SIZE, -1)
            data_batch = data_batch.to(device)
            y_pred = model(data_batch)
            loss = criterion(y_pred, labels)
            total_loss += loss.item()
            
            optimizer.zero_grad()

            loss.backward()

            # nn.utils.clip_grad_norm_(model.parameters(), 0) # gradient clipping

            optimizer.step()
            scheduler.step()

            if ( batch_id + 1 ) % checkpointing_freq == 0:
                train_batch = batch_id
                print('epoch {}, batch {}, train_loss: {}'.format(epoch, batch_id, total_loss/checkpointing_freq), flush = True)
                total_loss = 0
        
                model.eval()
                valid_loss = 0
                for batch_id, data_batch in enumerate(valid_dataloader):
                    if batch_id == int(len(valid_dataloader) * valid_used_ratio):
                        break
                    data_batch = data_batch.to(device)
                    y_pred = model(data_batch)
                    loss = criterion(y_pred, labels)
                    valid_loss += loss.item()

                valid_loss = valid_loss/int(len(valid_dataloader) * valid_used_ratio)
                print('epoch {}, batch {}, valid_loss: {}'.format(epoch, train_batch, valid_loss), flush = True)
                if valid_loss < lowest_loss:
                    lowest_loss = valid_loss
                    save_checkpoint(epoch, model, optimizer, valid_loss)

                model.train()
        
        print('end of epoch {}, latest validation loss: {}, best validation loss: {}'.format(epoch, valid_loss, lowest_loss))
        
            

            
           





    # attn_params = {
    #     # 'embed_dim': 768,
    #     'num_heads' : 4, 
    #     'dropout' : 0.2, 
    #     'intermediate_dimension' : 200,
    # }

    # batch = next(iter(train_dataloader))
    
    # model(batch, BATCH_SIZE)
    # print(model(next(iter(train_dataloader)), BATCH_SIZE).shape)

