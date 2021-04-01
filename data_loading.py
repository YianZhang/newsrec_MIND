import torch
import random
from transformers import BertTokenizer

class MINDDataset(torch.utils.data.Dataset):
  def __init__(self, news_file, behavior_file, npratio=5, his_size=5, col_spliter="\t", ID_spliter="%", 
  batch_size=1, title_size=50, model='bert-base-uncased', subset='train'):
    """ initialize the dataset. """
    self._titles = {}
    self._behaviors = []
    self._dataset = []

    #Note: the longest title in demo-train is 48.
    self.col_spliter = col_spliter
    self.ID_spliter = ID_spliter
    self.batch_size = batch_size
    self.title_size = title_size
    self.his_size = his_size
    self.npratio = npratio
    self.model = model
    self.subset = subset
    
    self.news_file = news_file
    self.behavior_file = behavior_file

    self.tokenizer = BertTokenizer.from_pretrained(self.model)
  
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

  # def init_behaviors(self):
  #   """get behaviors from behaviors_file, parse and store the impression and the history."""
  #   with open(self.behavior_file, 'r') as f:
  #     line = f.readline()
  #     while line != '':
  #       uid, time, history, impr = line.strip("\n").split(self.col_spliter)[-4:]
  #       history = history.split()
  #       history = [0] * (self.his_size - len(history)) + history[:self.his_size]

  #       positive, negative = [], []
  #       for news in impr.split():
  #         nid, label = news.split("-")
  #         if int(label):
  #           positive.append(nid)
  #         else:
  #           negative.append(nid)
      
  #       instance = {'history':history, 'positive': positive, 'negative': negative}
  #       self._behaviors.append(instance)
        
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
        uid, time, history, impr = line.strip("\n").split(self.col_spliter)[-4:]
        history = history.split()
        history = [0] * (self.his_size - len(history)) + history[:self.his_size]
        history = ['' if hid == 0 else self._titles[hid] for hid in history] # to be further discussed
        history_mask = [1 if his!='' else 0 for his in history]
        pos, neg = [], [] 
        # get the positive and negative ids in this impression
        if self.subset == 'train':
          for news in impr.split():
            nid, label = news.split("-")
            if int(label):
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
            self._dataset.append(instance)
        else:
          pass #test set: TODO

        line = f.readline()
  
  def __len__(self):
    return len(self._dataset)

  def __getitem__(self, i):
    return self._dataset[i]

  def collate_fn(self, batch):
    #Bertify
    #TODO: test set
    sentences = []
    for instance in batch:
      sentences+=instance['candidates']+instance['history']
    output = self.tokenizer(sentences, return_tensors="pt", padding = "longest")
    output['labels'] = torch.Tensor(([1] + [0] * self.npratio + [-1] * self.his_size) * len(batch))
    output['candidate_mask'] = torch.Tensor([instance['candidate_mask'] for instance in batch])
    output['history_mask'] = torch.Tensor([instance['history_mask'] for instance in batch])
    return output


if __name__ == '__main__':
  from torch.utils.data import RandomSampler
  from torch.utils.data import DataLoader

  my_ds = MINDDataset('train/news.tsv', 'train/behaviors.tsv',npratio=2, his_size=2, batch_size=4)
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
    
