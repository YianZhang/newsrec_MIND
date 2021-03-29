import torch
class Config:
  def __init__(self, entries):
    self._dict = entries
  def __getattr__(self,name):
      return self._dict[name]

def save_checkpoint(epoch, model, optimizer, loss, path='model.pt'):
  torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)
  print('checkpoint saved!')

def load_checkpoint(model, optimizer, map_location, path='model.pt'):
  checkpoint = torch.load(path, map_location=map_location)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  loss = checkpoint['loss']
  print('checkpoint loaded, epoch: {}, loss: {}'.format(epoch, loss))
  return epoch, loss