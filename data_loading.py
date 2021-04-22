import torch
import random
from transformers import BertTokenizer

class MINDDataset(torch.utils.data.Dataset):
  def __init__(self, news_file, behavior_file, npratio=4, his_size=50, col_spliter="\t", ID_spliter="%", 
  batch_size=1, title_size=50, model='bert-base-uncased', subset='train'):
    """ initialize the dataset. """
    self._titles = {}
    self._behaviors = []
    self._dataset = []
    self._title_reprs = {}
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

    self.tokenizer = BertTokenizer.from_pretrained(self.model) # assuming the model type is bert
  
  def init_titles(self):
    """ get news titles from news_file."""
    with open(self.news_file, 'r') as f:
      line = f.readline()
      while line != '':
        nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split(self.col_spliter)
        if nid in self._titles:
          continue
        self._titles[nid] = title
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
    self.init_titles()
    print('init titles finished')
    with open(self.behavior_file, 'r') as f:
      line = f.readline()
      while line != '':
        # get the histories
        # impr_id = line.strip("\n").split(self.col_spliter)[0] # for debugging
        uid, time, history, impr = line.strip("\n").split(self.col_spliter)[-4:]
        history = history.split()
        history = [0] * (self.his_size - len(history)) + history[:self.his_size]
        # hid = history # for debugging
        history = ['' if hid == 0 else self._titles[hid] for hid in history] # to be further discussed
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
            candidates = [self._titles[pid]] + [self._titles[nid] if nid!=0 else '' for nid in neg_samples] # to be further discussed
            candidate_mask = [1 if candidate!='' else 0 for candidate in candidates]
            #Note: uid not parsed since not used in our vanilla model.
            instance = {'history': history, 'candidates': candidates, 'history_mask': history_mask, 'candidate_mask': candidate_mask}
            # instance = {'history': history, 'candidates': candidates, 'history_mask': history_mask, 'candidate_mask': candidate_mask, 'impr_id': impr_id, 'pid': pid, 'nid': neg_samples, 'hid':hid} # for debugging
            self._dataset.append(instance)
        else:
          pass #test set: TODO

        line = f.readline()
    
  def encode_all_titles(self, title_encoder, batch_size = 128):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    title_encoder = title_encoder.to(device)
    if not hasattr(self, '_titles'):
      self.init_titles()
    indices, reprs = [], []
    i = 0
    batch_indices, batch_titles = [], []
    for nid, title in self._titles.items():
      batch_indices.append(nid)
      batch_titles.append(title)
      i += 1
      if i % batch_size == 0:
        #print('batch', i//batch_size, flush=True)
        encoder_input = self.tokenizer(batch_titles, return_tensors="pt", padding = "longest").to(device) #tokenize
        batch_reprs = title_encoder(**encoder_input).pooler_output.data.to('cpu') #forward
        indices.extend(batch_indices)
        reprs.extend(batch_reprs)
        #print(len(reprs), len(reprs[0]))
        batch_indices, batch_titles = [], [] # clear

    # forward and extend the rest titles
    encoder_input = self.tokenizer(batch_titles, return_tensors="pt", padding = "longest").to(device) #tokenize
    batch_reprs = title_encoder(**encoder_input).pooler_output.data.to('cpu') #forward
    indices.extend(batch_indices)
    reprs.extend(batch_reprs)
    self._title_reprs = dict(zip(indices, reprs))


  def load_data_for_evaluation(self): # under construction

    assert self.subset == 'valid' or 'test' # only the validation and the test sets are legal

    if not hasattr(self, '_titles'):
      self.init_titles()
    
    if not hasattr(self, '_title_reprs'):
      Exception("Please encode the titles first!")
    
    self._processed_impressions = []
    with open(self.behavior_file, 'r') as f:
      line = f.readline()
      while line != '':
        instance = {}
        # get the histories
        uid, time, history, impr = line.strip("\n").split(self.col_spliter)[-4:]
        history = history.split()
        history = [0] * (self.his_size - len(history)) + history[:self.his_size]
        #history_reprs = [torch.zeros(768) if hid == 0 else self._title_reprs[hid] for hid in history]
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

  def get_predictions(self, model):
    if self.subset == 'valid':
      labels, preds = [], []
      for instance in self._processed_impressions:
        instance['candidate_reprs'] = [self._title_reprs[nid] for nid in instance['candidates']]
        instance['history_reprs'] = [torch.zeros(768) if hid == 0 else self._title_reprs[hid] for hid in instance['history_ids']
        labels.append(self._processed_impressions['labels'])
        preds.append(model.predict(instance))
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
    sentences = []
    # impr_ids = [] # for debugging
    # pids = [] # for debugging
    # nids = [] # for debugging
    # hids = [] # for debugging
    for instance in batch:
      sentences+=instance['candidates']+instance['history']
      # impr_ids.append(instance['impr_id']) # for debugging
      # pids.append(instance['pid']) # for debugging
      # nids.append(instance['nid']) # for debugging
      # hids.append(instance['hid']) # for debugging
    output = self.tokenizer(sentences, return_tensors="pt", padding = "longest")
    # output['sentences'] = sentences # for debugging
    # output['impr_ids'] = impr_ids # for debugging
    # output['pids'] = pids # for debugging
    # output['nids'] = nids # for debugging
    # output['hids'] = hids # for debugging
    output['labels'] = torch.Tensor(([1] + [0] * self.npratio + [-1] * self.his_size) * len(batch))
    output['candidate_mask'] = torch.Tensor([instance['candidate_mask'] for instance in batch])
    output['history_mask'] = torch.Tensor([instance['history_mask'] for instance in batch])
    return output


if __name__ == '__main__':
  from torch.utils.data import RandomSampler
  from torch.utils.data import DataLoader

  my_ds = MINDDataset('train/news.tsv', 'train/behaviors.tsv',npratio=2, his_size=2, batch_size=2)
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
    
