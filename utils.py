class Config:
  def __init__(self, entries):
    self._dict = entries
  def __getattr__(self,name):
      return self._dict[name]