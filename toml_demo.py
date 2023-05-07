import toml

class objdict(dict):
  def __getattr__(self, name):
    assert name in self, (name, self.keys())
    return self[name]

  def __setattr__(self, name, value):
    assert name in self, (name, self.keys())
    self[name] = value

res = toml.load('foo.toml', objdict)

print('title', res.title)
print('ar', res.c1.ar)
