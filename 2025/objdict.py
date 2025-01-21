
class objdict(dict):
  def __getattr__(self, name):
    assert name in self, (name, self.keys())
    return self[name]
