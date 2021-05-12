import math
import torch
from torch import nn
from transformers import AutoModel
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers import get_linear_schedule_with_warmup
from evaluate import evaluate
import random

# multi-GPU:
# Device 还是 cuda0. 还是to(device)
# optimizer, scheduler 弄好之后加torch.nn.DataParallel
# fp16

# 热度

# TODO List:
# most recent history + position embedding
# the init func of MySelfAttention
# 

class MySelfAttention(BertSelfAttention):
  # init: (self, config)
  def __init__(self, config):
        super().__init__(config)

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
            print(attention_scores.shape, self.convert_attention_mask(attention_mask).shape)
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

class Pseudo_MLP_Scorer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dropout = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim1, dim2)
        self.linear2 = nn.Linear(dim2, dim3)
        self.linear3 = nn.Linear(dim3, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        #return self.linear2(self.activation(self.dropout(self.linear1(x))))
        return self.linear3(self.activation(self.dropout(self.linear2(self.activation(self.dropout(self.linear1(x)))))))
    
    def score(self, candidate_reprs, user_reprs):
        return self.forward(torch.cat((candidate_reprs, user_reprs.view(user_reprs.shape[0],1,user_reprs.shape[1]).expand((-1,candidate_reprs.shape[1],-1))), dim=2))

class Dot_product_scorer(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, candidate_reprs, user_reprs):
        return torch.bmm(candidate_reprs, user_reprs.view(user_reprs.shape[0], user_reprs.shape[1], 1))
    def score(self, candidate_reprs, user_reprs):
        return self.forward(candidate_reprs, user_reprs)

class News_encoder(torch.nn.Module):
    def __init__(self, ht_model, news_encoder_parameters):
        super().__init__()
        self.ht_model = ht_model
        self.title_encoder = AutoModel.from_pretrained(self.ht_model)
        self.abstract_encoder = AutoModel.from_pretrained(self.ht_model)
        self.class_embedding = torch.nn.Embedding(news_encoder_parameters['n_classes'], news_encoder_parameters['class_embedding_dim'])
        self.subclass_embedding = torch.nn.Embedding(news_encoder_parameters['n_subclasses'], news_encoder_parameters['subclass_embedding_dim'])
        self.class_dropout = nn.Dropout(news_encoder_parameters['class_dropout'])
        self.distil_dropout = nn.Dropout(news_encoder_parameters['distil_dropout'])
        self.distil = nn.Linear(news_encoder_parameters['class_embedding_dim'] + 
                                news_encoder_parameters['subclass_embedding_dim'] + 
                                news_encoder_parameters['entity_embedding_dim'] * 2 + 
                                self.title_encoder.config.hidden_size * 2, 
                                    news_encoder_parameters['news_repr_dim'])

    def forward(self, x):
        # classes
        class_embeddings, subclass_embeddings = self.class_dropout(self.class_embedding(x['classes'])), self.class_dropout(self.subclass_embedding(x['subclasses']))
        
        # todo title_entity_embeddings, abstract_entity_embeddings = x['title_entity_embeddings'], x['abstract_entity_embeddings']
        #del x['title_entity_embeddings']; del x['abstract_entity_embeddings']

        if self.ht_model == 'bert-base-uncased' or self.ht_model.startswith('prajjwal1/bert'):
            title_inputs= x['titles']
            for key, val in title_inputs.items():
                shape = val.shape
                title_inputs[key] = val.view((-1, shape[-1]))
            title_reprs = self.title_encoder(**(x['titles'])).pooler_output.view((shape[0],shape[1],-1))

            abstract_inputs= x['abstracts']
            for key, val in abstract_inputs.items():
                shape = val.shape
                abstract_inputs[key] = val.view((-1, shape[-1]))
            abstract_reprs = self.abstract_encoder(**(x['abstracts'])).pooler_output.view((shape[0], shape[1], -1))
        elif self.ht_model == 'distilbert-base-uncased':
            # not implemented for distilbert
            title_reprs = self.title_encoder(**(x['titles'])).last_hidden_state[:,0,:].flatten()
            abstract_reprs = self.abstract_encoder(**(x['abstracts'])).last_hidden_state[:,0,:].flatten()
        # debuggin:
        for tensor in (title_reprs, abstract_reprs, class_embeddings, subclass_embeddings, x['title_entity_embeddings'], x['abstract_entity_embeddings']):
            print(tensor.shape)
        catted = torch.cat((title_reprs, abstract_reprs, class_embeddings, subclass_embeddings, x['title_entity_embeddings'], x['abstract_entity_embeddings']), dim=-1)
        return self.distil(self.distil_dropout(catted))

class NewsRec(torch.nn.Module):
    def __init__(self, self_attention_config, news_encoder_parameters, ht_model='bert-base-uncased', scorer = 'dot_product'):
        super().__init__()
        self.news_encoder_parameters = news_encoder_parameters
        self.ht_model = ht_model
        self.news_encoder = News_encoder(ht_model, self.news_encoder_parameters)
        self.news_MHA = MySelfAttention(self_attention_config)
        self.news_pooling = Attention_pooling(self.news_encoder_parameters['news_repr_dim'])
        if scorer == 'dot_product':
            self.scorer = Dot_product_scorer()
        elif scorer == 'pseudo_MLP':
            self.scorer = Pseudo_MLP_Scorer(self.news_encoder_parameters['news_repr_dim']*2, self.news_encoder_parameters['news_repr_dim'], self.news_encoder_parameters['news_repr_dim']//2)

    def masking(self, x, candidate_mask):
        mask = ((1 - candidate_mask) * -10000.0).to(dtype=next(self.parameters()).dtype)
        return x + mask

    # def dot_product_score(self, candidate_reprs, user_reprs):
    #     return torch.bmm(candidate_reprs, user_reprs.view(user_reprs.shape[0], user_reprs.shape[1], 1))

    # def pseudo_MLP_score(self, candidate_reprs, user_reprs):
    #     return self.pseudo_MLP_scorer(torch.cat((candidate_reprs, user_reprs.view(user_reprs.shape[0],1,user_reprs.shape[1]).expand((-1,candidate_reprs.shape[1],-1))), dim=2))

    def predict(self, instance):
        hr_shape, hm_shape, cr_shape = instance['history_reprs'].shape, instance['history_mask'].shape, instance['candidate_reprs'].shape # get shape for reshaping later
        history_reprs = instance['history_reprs'].view(1, hr_shape[0], hr_shape[1]) # reshape to match the input format
        history_mask = instance['history_mask'].view(1, hm_shape[0])
        candidate_reprs = instance['candidate_reprs'].view(1, cr_shape[0], cr_shape[1])
        self_attended_his_reprs = self.news_MHA(history_reprs, history_mask)[0] # self-attention
        user_reprs = self.news_pooling(self_attended_his_reprs, history_mask) # attention pooling
        scores = self.scorer.score(candidate_reprs, user_reprs).data.flatten().to('cpu') # dot product
        return scores

    def forward(self, x):
        batch_size = x['candidate_mask'].shape[0]

        his_indices = torch.nonzero(x['labels'] == -1, as_tuple = True)[0]
        impr_indices = torch.nonzero(x['labels'] != -1, as_tuple = True)[0]
        
        news_reprs = self.news_encoder(x)

        # his_reprs = news_reprs[his_indices].view(batch_size, -1, news_reprs.shape[-1])
        his_reprs = news_reprs[his_indices]
        
        # impr_labels = labels[impr_indices].view(batch_size, -1)
        
        # impr_reprs = news_reprs[impr_indices].view(batch_size, -1, news_reprs.shape[-1])
        impr_reprs = news_reprs[impr_indices]

        self_attended_his_reprs = self.news_MHA(his_reprs, x['history_mask'])[0]

        user_reprs = self.news_pooling(self_attended_his_reprs, x['history_mask'])

        # print(user_reprs.shape, impr_reprs.shape) # checked. shapes are correct

        scores = self.scorer.score(impr_reprs, user_reprs).squeeze(-1)

        # print(user_reprs[1], impr_reprs[1][1], scores) # checked. bmm is correct.

        return self.masking(scores, x['candidate_mask'])


if __name__ == '__main__':

    torch.manual_seed(42) # for reproducibility
    random.seed(42) # for reproducibility

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_name', default = 'model.pt')
    parser.add_argument('--lr', default = 3e-5, type = float)
    parser.add_argument('--pretrained_model', default = 'bert-base-uncased')
    parser.add_argument('--datasize', default = 'demo')
    parser.add_argument('--warmup_ratio', type = float, default = 0.06)
    parser.add_argument('--attn_dropout', type = float, default = 0.2)
    parser.add_argument('--patience', type = int, default = -1)
    parser.add_argument('--scorer', default = 'dot_product')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device, flush = True)

    # model HPs
    # position embedding related HPs are useless.
    if args.pretrained_model == 'bert-base-uncased':
        BATCH_SIZE = 3 * torch.cuda.device_count() # 6 works for demo, not for large
        # HIDDEN_SIZE = 768
    elif args.pretrained_model == 'distilbert-base-uncased':
        BATCH_SIZE = 8 * torch.cuda.device_count()# 12 does not work for small
        # HIDDEN_SIZE = 768
    elif args.pretrained_model == 'prajjwal1/bert-tiny':
        BATCH_SIZE = 24 * torch.cuda.device_count()
        # HIDDEN_SIZE = 128
    elif args.pretrained_model == 'prajjwal1/bert-mini':
        BATCH_SIZE = 16 * torch.cuda.device_count()
        # HIDDEN_SIZE = 256
        
    self_attention_hyperparameters = {'num_attention_heads' : 20, 'attention_probs_dropout_prob': args.attn_dropout, 'max_position_embeddings': 4, 'is_decoder': False, 'position_embedding_type' : None,}
    # assert self_attention_hyperparameters['hidden_size'] % self_attention_hyperparameters['num_attention_heads'] == 0
    # get data
    from data_loading import MINDDataset
    from torch.utils.data import RandomSampler
    from torch.utils.data import DataLoader
    from utils import Config, save_checkpoint, load_checkpoint
    from os import path

    DATA_SIZE = args.datasize # demo, small, large
    train = MINDDataset(path.join(DATA_SIZE,'train/news.tsv'), path.join(DATA_SIZE,'train/behaviors.tsv'), 'all_embeddings.vec', 'large', batch_size=BATCH_SIZE, model=args.pretrained_model)
    train.load_data()
    train_sampler = RandomSampler(train)
    train_dataloader = DataLoader(
    train,
    sampler=train_sampler,
    batch_size=train.batch_size,
    collate_fn=train.collate_fn
    )

    valid = MINDDataset(path.join(DATA_SIZE,'valid/news.tsv'), path.join(DATA_SIZE,'valid/behaviors.tsv'), 'all_embeddings.vec', 'large', batch_size=BATCH_SIZE, model=args.pretrained_model, subset='valid')
    valid.load_data()
    valid_sampler = RandomSampler(valid)
    valid_dataloader = DataLoader(
    valid,
    sampler=valid_sampler,
    batch_size=valid.batch_size,
    collate_fn=valid.collate_fn
    )

    print('checking the class2id matrices:', train._class2id == valid._class2id, train._subclass2id == valid._subclass2id)
    print('finish loading data', flush = True)

    # build the model
    news_encoder_parameters = {'n_classes': len(train._class2id), 'n_subclasses': len(train._subclass2id), 'class_embedding_dim': 50, 'subclass_embedding_dim': 30, 'news_repr_dim': 100, 'distil_dropout': 0.1, 'class_dropout': 0, 'entity_embedding_dim': 100}
    self_attention_hyperparameters['hidden_size'] = news_encoder_parameters['news_repr_dim']
    self_attention_config = Config(self_attention_hyperparameters)
    model = NewsRec(self_attention_config, news_encoder_parameters, args.pretrained_model, args.scorer).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    print('finish building the model', flush = True)


    # training
    # BERT parameters:
    # batch size: 16, 32
    # lr(Adam): 5e-5, 3e-5, 2e-5
    # number of epochs: 2,3,4
    # dropout: 0.1

    lr = args.lr
    # num_warmup_steps = args.warmup_steps # bert 10,000 # I used 3000 for demo
    checkpointing_freq = 250 # for demo I used 200

    valid_metrics_ratio = (0.5, 1)
    if args.datasize == 'demo':
        valid_loss_ratio = 0.02 # demo # out of 716 * 16 # for demo I used 0.02
        MAX_EPOCHS = 7
    elif args.datasize == 'small':
        valid_loss_ratio = 0.003 # small # out of 6962 * 16
        MAX_EPOCHS = 3
    elif args.datasize == 'large':
        valid_loss_ratio = 0.0005 
        MAX_EPOCHS = 1
        valid_metrics_ratio = (0.1, 0.2)
    

    num_train_steps = MAX_EPOCHS*(len(train_dataloader) - 1) # to be further checked
    num_warmup_steps = num_train_steps * args.warmup_ratio
    print("num_train_steps: {}, num_warmup_steps: {}".format(num_train_steps, num_warmup_steps))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
    criterion = torch.nn.CrossEntropyLoss() # default reduction = "mean"
    labels = torch.tensor([0] * BATCH_SIZE).to(device)
    
    try:
        load_checkpoint(model, optimizer, device, args.checkpoint_name)
        print('checkpoint loaded', flush = True)
    except:
        print('failed to load any checkpoints.', flush = True)

    best_performance = -100000
    KEY_METRIC = "group_auc"
    no_improvement = 0
    early_stop_now = False

    for epoch in range(MAX_EPOCHS):
        if early_stop_now:
            break
        total_loss = 0
        model.train()

        print('start training ...', flush = True)
        for batch_id, data_batch in enumerate(train_dataloader):
            if batch_id == len(train_dataloader) - 1:
                continue
            # impr_indices = torch.nonzero(data_batch["labels"] != -1, as_tuple = True)[0] # must retrieve the labels first, because it is deleted by the forward func
            # impr_labels = data_batch["labels"][impr_indices].view(BATCH_SIZE, -1)
            for key in data_batch:
                data_batch[key] = data_batch[key].to(device)

            y_pred = model(data_batch)
            loss = criterion(y_pred, labels)
            total_loss += loss.item()
            
            optimizer.zero_grad()

            loss.backward()

            # nn.utils.clip_grad_norm_(model.parameters(), 1) # gradient clipping

            optimizer.step()
            scheduler.step()

            if ( batch_id + 1 ) % checkpointing_freq == 0:
                print('epoch {}, batch {}/{}, train_loss: {}'.format(epoch, batch_id, len(train_dataloader), total_loss/checkpointing_freq), flush = True)
                total_loss = 0

                print('validating...', flush = True)
                model.eval()
                valid_loss = 0
                for batch_id, data_batch in enumerate(valid_dataloader):
                    if batch_id == int(len(valid_dataloader) * valid_loss_ratio):
                        break
                    for key in data_batch:
                        data_batch[key] = data_batch[key].to(device)
                    y_pred = model(data_batch)
                    loss = criterion(y_pred, labels)
                    valid_loss += loss.item()

                valid_loss = valid_loss/int(len(valid_dataloader) * valid_loss_ratio)
                print('valid_loss: {}'.format(valid_loss), flush = True)
                print('learning rates:', flush = True)
                for param_group in optimizer.param_groups:
                    print(param_group['lr'], end = ' ', flush = True)

                evaluation_metrics = evaluate(valid, model, valid_metrics_ratio[0])
                print(evaluation_metrics, flush = True)
                key_metric = evaluation_metrics[KEY_METRIC]
                if key_metric > best_performance:
                    best_performance = key_metric
                    save_checkpoint(epoch, model, optimizer, valid_loss, args.checkpoint_name)
                    no_improvement = 0
                else:
                    no_improvement +=1
                    if no_improvement > args.patience > 0:
                        early_stop_now = True
                        break

                print("Best {} score so far: {}, no improvement for {} updates".format(KEY_METRIC, best_performance, no_improvement), flush = True)
                model.train()

        print('end of epoch validating...', flush = True)
        model.eval()
        valid_loss = 0
        for batch_id, data_batch in enumerate(valid_dataloader):
            if batch_id == int(len(valid_dataloader) * 0.1):
                break
            for key in data_batch:
                data_batch[key] = data_batch[key].to(device)
            y_pred = model(data_batch)
            loss = criterion(y_pred, labels)
            valid_loss += loss.item()

        valid_loss = valid_loss/int(len(valid_dataloader) * 0.1)

        print('end of epoch {}, full validation set loss: {}'.format(epoch, valid_loss), flush = True)
        print(evaluate(valid, model, valid_metrics_ratio[1]), flush = True)

    if early_stop_now:
        print('The training is terminated by early stopping.', flush = True)
        
            

            
           





    # attn_params = {
    #     # 'embed_dim': 768,
    #     'num_heads' : 4, 
    #     'dropout' : 0.2, 
    #     'intermediate_dimension' : 200,
    # }

    # batch = next(iter(train_dataloader))
    
    # model(batch, BATCH_SIZE)
    # print(model(next(iter(train_dataloader)), BATCH_SIZE).shape)

