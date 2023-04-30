import toml
from absl import app

class objdict(dict):
  def __getattr__(self, name):
    assert name in self, (name, self.keys())
    return self[name]


def _fix_defaults(plan):
  return plan


def load_plan(fn):
  return _fix_defaults(toml.load(fn, objdict))


def main(argv):
  print(load_plan('v0.toml'))


if __name__ == '__main__':
  app.run(main)
