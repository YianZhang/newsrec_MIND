import torch
import numpy as np
import random
from transformers import BertTokenizer, DistilBertTokenizer
from utils import get_class_dictionaries

class MINDDataset(torch.utils.data.Dataset):
  def __init__(self, news_file, behavior_file, entity_file, large_address, npratio=4, his_size=50, col_spliter="\t", ID_spliter="%", 
  batch_size=1, title_size=50, model='bert-base-uncased', subset='train'):
    """ initialize the dataset. """
    self._titles, self._abstracts, self._bodies = {}, {}, {} # body data not available at this point
    self._classes = {}
    self._class2id, self._subclass2id = get_class_dictionaries(large_address, col_spliter)
    self._entity_embeddings = {}
    self._news_entity_embeddings = {}
    self._dataset = []
    self._news_reprs = {}
    self._processed_impressions = []

    #Note: the longest title in demo-train is 48.
    self.col_spliter = col_spliter
    self.ID_spliter = ID_spliter
    self.batch_size = batch_size
    self.title_size = title_size # not used so far
    self.his_size = his_size
    self.npratio = npratio
    self.model = model
    self.subset = subset
    
    self.news_file = news_file
    self.behavior_file = behavior_file
    self.entity_file = entity_file

    if model.startswith('bert') or self.model.startswith('prajjwal1/bert'):
      self.tokenizer = BertTokenizer.from_pretrained(self.model)
    elif model.startswith('distilbert'):
      self.tokenizer = DistilBertTokenizer.from_pretrained(self.model)
  
  def init_entities(self):
    for line in open(self.entity_file, 'r').readlines():
      line_entries = line.strip().split('\t')
      entity_id = line_entries[0]
      entity_embedding = [float(line_entries[i]) for i in range(1, len(line_entries))]
      self._entity_embeddings[entity_id] = entity_embedding
  
  def init_news(self):
    """ get news titles and classes from news_file."""
    if self._titles != {}:
      print('Warning: Overwriting the loaded titles')
      self._titles = {}
    if self._abstracts != {}:
      print('Warning: Overwriting the loaded abstracts')
      self._abstracts = {}
    if self._entity_embeddings == {}:
      self.init_entities()

    with open(self.news_file, 'r') as f:
      line = f.readline()
      while line != '':
        nid, vert, subvert, title, ab, url, t_entities, a_entities = line.strip("\n").split(self.col_spliter)
        if nid in self._titles:
          continue
        self._titles[nid] = title
        self._abstracts[nid] = ab
        self._classes[nid] = (vert, subvert)
        
        # entities
        t_all = torch.tensor([self._entity_embeddings[entity['WikidataId']] for entity in eval(t_entities) if entity['WikidataId'] in self._entity_embeddings ])
        if len(t_all) == 0:
          t_emtity_embedding = torch.tensor(self._entity_embeddings['average'])
        else:
          t_entity_embedding = torch.mean(t_all, dim = 0)
        
        a_all = torch.tensor([self._entity_embeddings[entity['WikidataId']] for entity in eval(a_entities) if entity['WikidataId'] in self._entity_embeddings ])
        if len(a_all) == 0:
          a_entity_embedding = torch.tensor(self._entity_embeddings['average'])
        else:
          a_entity_embedding = torch.mean(a_all, dim = 0)

        self._news_entity_embeddings[nid] = (t_entity_embedding, a_entity_embedding)

        line = f.readline()
        
  def newsample(self, news, ratio):
    """ Sample ratio samples from news list. 
    If length of news is less than ratio, pad zeros.

    Args:
        news (list): input news list
        ratio (int): sample number
    
    Returns:
        list: output of sample list.
    """
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)
   

  def load_data(self):
    self.init_entities()
    self.init_news()
    print('init news finished')
    with open(self.behavior_file, 'r') as f:
      line = f.readline()
      while line != '':
        # get the histories
        # impr_id = line.strip("\n").split(self.col_spliter)[0] # for debugging
        uid, time, his_ids, impr = line.strip("\n").split(self.col_spliter)[-4:]
        his_ids = his_ids.split()
        his_ids = [0] * (self.his_size - len(his_ids)) + his_ids[-self.his_size:]
        # hid = history # for debugging
        history_titles = ['' if hid == 0 else self._titles[hid] for hid in his_ids]
        history_abstracts = ['' if hid == 0 else self._abstracts[hid] for hid in his_ids]
        history_classes = [('','') if hid == 0 else self._classes[hid] for hid in his_ids]
        history_entity_embeddings = [(torch.tensor(self._entity_embeddings['average']), torch.tensor(self._entity_embeddings['average'])) if hid == 0 else self._news_entity_embeddings[hid] for hid in his_ids]
        history_mask = [1 if his!='' else 0 for his in history]
        pos, neg = [], [] 
        # get the positive and negative ids in this impression
        if self.subset == 'train' or 'valid':
          for news in impr.split():
            nid, label = news.split("-")
            if label == '1':
              pos.append(nid)
            else:
              neg.append(nid)
          
          # make an instance for each positive sample in the impression
          for pid in pos:
            neg_samples = self.newsample(neg, self.npratio)
            candidate_titles = [self._titles[pid]] + [self._titles[nid] if nid!=0 else '' for nid in neg_samples]
            candidate_abstracts = [self._abstracts[pid]] + [self._abstracts[nid] if nid!=0 else '' for nid in neg_samples]
            candidate_classes = [self._classes[pid]] + [self._classes[nid] if nid!=0 else ('','') for nid in neg_samples]
            candidate_entity_embeddings = [self._news_entity_embeddings[pid]] + [self._news_entity_embeddings[nid] if nid!=0 else (torch.tensor(self._entity_embeddings['average']), torch.tensor(self._entity_embeddings['average'])) for nid in neg_samples]
            candidate_mask = [1 if candidate!='' else 0 for candidate in candidate_titles]
            #Note: uid not parsed since not used in our vanilla model.
            instance = {'history': history, 'candidate_titles': candidate_titles, 'candidate_abstracts': candidate_abstracts, 'history_classes': history_classes, 'candidate_classes': candidate_classes, 'history_mask': history_mask, 'candidate_mask': candidate_mask, 'candidate_entity_embeddings': candidate_entity_embeddings, 'history_entity_embeddings': history_entity_embeddings}
            # instance = {'history': history, 'candidates': candidates, 'history_mask': history_mask, 'candidate_mask': candidate_mask, 'impr_id': impr_id, 'pid': pid, 'nid': neg_samples, 'hid':hid} # for debugging
            self._dataset.append(instance)
        else:
          pass #test set: TODO

        line = f.readline()
    
  def encode_all_news(self, news_encoder, batch_size = 128):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    news_encoder = news_encoder.to(device)
    if self._titles == {}:
      self.init_news()
    indices, reprs = [], []
    i = 0
    batch_indices, batch_titles, batch_abstracts, batch_classes, batch_subclasses, batch_title_entity_embeddings, batch_abstract_entity_embeddings = [], [], [], [], [], [], []
    for nid, title in self._titles.items():
      batch_indices.append(nid)
      batch_titles.append(title)
      batch_abstracts.append(self._abstracts[nid])
      vert, subvert = self._classes[nid]
      batch_classes.append(self._class2id[vert])
      batch_subclasses.append(self._subclass2id[subvert])
      title_entity_embedding, abstract_entity_embedding = self._news_entity_embeddings[nid]
      batch_title_entity_embeddings.append(title_entity_embedding)
      batch_abstract_entity_embeddings.append(abstract_entity_embedding)
      
      i += 1
      if i % batch_size == 0:
        #print('batch', i//batch_size, flush=True)
        encoder_input = {}
        encoder_input['titles'] = self.tokenizer(batch_titles, return_tensors="pt", padding = "longest") #tokenize
        encoder_input['abstracts'] = self.tokenizer(batch_abstracts, return_tensors="pt", padding = "longest") #tokenize
        encoder_input['classes'] = torch.LongTensor(batch_classes)
        encoder_input['subclasses'] = torch.LongTensor(batch_subclasses)
        encoder_input['title_entity_embeddings'] = torch.stack(batch_title_entity_embeddings)
        encoder_input['abstract_entity_embeddings'] = torch.stack(batch_abstract_entity_embeddings)
        batch_reprs = news_encoder(encoder_input.to(device)).data.to('cpu')
        # if self.model == 'bert-base-uncased' or self.model.startswith('prajjwal1/bert'):
        #   batch_reprs = news_encoder(**encoder_input).pooler_output.data.to('cpu') #forward
        # elif self.model == 'distilbert-base-uncased':
        #   batch_reprs = news_encoder(**encoder_input).last_hidden_state[:,0,:].data.to('cpu') #forward

        indices.extend(batch_indices)
        reprs.extend(batch_reprs)
        #print(len(reprs), len(reprs[0]))
        batch_indices, batch_titles, batch_abstracts, batch_classes, batch_subclasses, batch_title_entity_embeddings, batch_abstract_entity_embeddings = [], [], [], [], [], [], [] # clear

    # forward and extend the rest titles
    encoder_input = {}
    encoder_input['titles'] = self.tokenizer(batch_titles, return_tensors="pt", padding = "longest") #tokenize
    encoder_input['abstracts'] = self.tokenizer(batch_abstracts, return_tensors="pt", padding = "longest") #tokenize
    encoder_input['classes'] = torch.LongTensor(batch_classes)
    encoder_input['subclasses'] = torch.LongTensor(batch_subclasses)
    encoder_input['title_entity_embeddings'] = torch.stack(batch_title_entity_embeddings)
    encoder_input['abstract_entity_embeddings'] = torch.stack(batch_abstract_entity_embeddings)
    batch_reprs = news_encoder(encoder_input.to(device)).data.to('cpu')
    
    indices.extend(batch_indices)
    reprs.extend(batch_reprs)
    self._news_reprs = dict(zip(indices, reprs))


  def load_data_for_evaluation(self): # under construction

    assert self.subset == 'valid' or 'test' # only the validation and the test sets are legal

    if self._processed_impressions != []:
      print('Impressions already processed.', flush = True)
      return

    if self._titles == {} or self._classes == {}:
      self.init_news()
    
    if not hasattr(self, '_news_reprs'):
      Exception("Please encode the titles first!")
    
    self._processed_impressions = []
    with open(self.behavior_file, 'r') as f:
      line = f.readline()
      while line != '':
        instance = {}
        # get the histories
        uid, time, history, impr = line.strip("\n").split(self.col_spliter)[-4:]
        history = history.split()
        history = [0] * (self.his_size - len(history)) + history[-self.his_size:]
        history_mask = [1 if hid!=0 else 0 for hid in history]

        ids, labels = [], []
        if self.subset == 'valid':
          for news in impr.split():
            nid, label = news.split("-")
            ids.append(nid)
            labels.append(int(label))
          
          #candidate_reprs = [self._title_reprs[nid] for nid in ids]
          # candidate_mask = [1 if candidate!='' else 0 for candidate in candidates] # don't think we need this
          
          instance = {'history_ids': history, 'history_mask': history_mask, 'candidates': ids, 'labels': labels}
          #instance = {'history_reprs': history_reprs, 'history_mask': history_mask, 'candidate_reprs': candidate_reprs, 'labels': labels}
          self._processed_impressions.append(instance)
        else:
          pass #test set: TODO

        line = f.readline()

  def get_predictions(self, model, ratio=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    if self.subset == 'valid':
      labels, preds = [], []  
      for instance in self._processed_impressions[:int(ratio*len(self._processed_impressions))]:
        instance['candidate_reprs'] = torch.stack([self._news_reprs[nid] for nid in instance['candidates']]).to(device)
        instance['history_reprs'] = torch.stack([torch.zeros(model.news_encoder_parameters['news_repr_dim']) if hid == 0 else self._title_reprs[hid] for hid in instance['history_ids']]).to(device) #title_todo
        instance['history_mask'] = torch.tensor(instance['history_mask']).to(device)
        labels.append(np.array(instance['labels']))
        preds.append(model.predict(instance).numpy())
      return labels, preds
    else:
      pass
  
  def __len__(self):
    return len(self._dataset)

  def __getitem__(self, i):
    return self._dataset[i]

  def collate_fn(self, batch):
    #Bertify
    #TODO: test set
    titles, abstracts = [], []
    classes = []
    news_entity_embeddings = []
    output = {}
    # impr_ids = [] # for debugging
    # pids = [] # for debugging
    # nids = [] # for debugging
    # hids = [] # for debugging
    for instance in batch:
      titles += instance['candidate_titles']+instance['history_titles']
      abstracts += instance['candidate_abstracts']+instance['history_abstracts']
      classes += instance['candidate_classes']+instance['history_classes']
      news_entity_embeddings += instance['candidate_entity_embeddings'] + instance['history_entity_embeddings']
      # impr_ids.append(instance['impr_id']) # for debugging
      # pids.append(instance['pid']) # for debugging
      # nids.append(instance['nid']) # for debugging
      # hids.append(instance['hid']) # for debugging
    title_encodings = self.tokenizer(titles, return_tensors="pt", padding = "longest")
    abstract_encodings = self.tokenizer(abstracts, return_tensors="pt", padding = "longest")
    # output['sentences'] = sentences # for debugging
    # output['impr_ids'] = impr_ids # for debugging
    # output['pids'] = pids # for debugging
    # output['nids'] = nids # for debugging
    # output['hids'] = hids # for debugging
    output['titles'] = title_encodings
    output['abstracts'] = abstract_encodings
    output['labels'] = torch.Tensor(([1] + [0] * self.npratio + [-1] * self.his_size) * len(batch))
    output['candidate_mask'] = torch.Tensor([instance['candidate_mask'] for instance in batch])
    output['history_mask'] = torch.Tensor([instance['history_mask'] for instance in batch])
    # print(classes)
    output['classes'] = torch.LongTensor([ self._class2id[vert] for vert, _ in classes])
    output['subclasses'] = torch.LongTensor([ self._subclass2id[subvert] for _, subvert in classes])
    output['title_entity_embeddings'] = torch.stack([ t_e_e for t_e_e, _ in news_entity_embeddings])
    output['abstract_entity_embeddings'] = torch.stack([ a_e_e for _, a_e_e in news_entity_embeddings])
    return output


if __name__ == '__main__':
  torch.manual_seed(42)
  from torch.utils.data import RandomSampler
  from torch.utils.data import DataLoader

  my_ds = MINDDataset('demo/train/news.tsv', 'demo/train/behaviors.tsv', 'all_embeddings.vec', npratio=2, his_size=2, batch_size=2)
  my_ds.load_data()

  train_sampler = RandomSampler(my_ds)
  train_dataloader = DataLoader(
  my_ds,
  sampler=train_sampler,
  batch_size=my_ds.batch_size,
  collate_fn=my_ds.collate_fn
  )

  it = iter(train_dataloader)
  for i in range(1):
    print(next(it))
    
