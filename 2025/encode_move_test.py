"""Test encode_move."""

from absl.testing import absltest

from encode_move import INDEX_TO_MOVE, MOVE_TO_INDEX

class EncodeMoveTest(absltest.TestCase):
  def test_encode_move(self):
    self.assertEqual(len(INDEX_TO_MOVE), 1968)
    self.assertEqual(INDEX_TO_MOVE[10], 'a1c1')
    self.assertEqual(MOVE_TO_INDEX['a1c1'], 10)

if __name__ == '__main__':
  absltest.main()
