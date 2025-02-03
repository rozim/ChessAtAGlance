from absl.testing import absltest

from az_model import AzModel


class AzModelTest(absltest.TestCase):
  def test_print(self):
    m = AzModel()
    print(m)

  def test_with_encoded_board(self):
    m = AzModel()
    print(m)

if __name__ == '__main__':
  absltest.main( )
