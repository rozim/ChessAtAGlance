import toml

class objdict(dict):
  def __getattr__(self, name):
    assert name in self, (name, self.keys())
    return self[name]

res = toml.load('foo.toml', objdict)

print('title', res.title)
print('ar', res.c1.ar)
