import toml
from absl import app

class objdict(dict):
  def __getattr__(self, name):
    assert name in self, (name, self.keys())
    return self[name]


def load_plan(fn):
  return toml.load(fn, objdict)



def main(argv):
  print(load_plan('v0.toml'))
  print(load_plan('v0.toml').data)  
  

if __name__ == '__main__':
  app.run(main)
