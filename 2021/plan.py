import toml
from absl import app

class objdict(dict):
  def __getattr__(self, name):
    assert name in self, (name, self.keys())
    return self[name]


def _fix_defaults(plan):
  mplan = plan.model
  mplan.mask_legal_moves = mplan.get('mask_legal_moves', False)
  assert 'num_cnn' not in mplan, 'Obsolete'
  mplan.do_squeeze_excite = mplan.get('do_squeeze_excite', False)

  for k, v in { 'squeeze_excite_ratio': 16}.items():
    if k not in mplan:
      mplan[k] = v

  dplan = plan.data
  dplan.prefetch_to_device = dplan.get('prefetch_to_device', False)

  return plan


def load_plan(fn):
  return _fix_defaults(toml.load(fn, objdict))



def main(argv):
  print(load_plan('v0.toml'))
  print(load_plan('v0.toml').data)


if __name__ == '__main__':
  app.run(main)
