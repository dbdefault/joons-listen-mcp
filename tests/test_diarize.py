import importlib.util
import unittest

try:
    import numpy as np
    from joons_listen import diarize
except Exception:  # numpy 없음
    np = None
    diarize = None

HAS_SK = importlib.util.find_spec("sklearn") is not None


@unittest.skipUnless(np is not None, "numpy 없음")
class MakeWindowsTest(unittest.TestCase):
    def test_short_segment_single_window(self):
        self.assertEqual(diarize.make_windows([(0.0, 4.0)], win=3.0), [(0.0, 4.0, 0)])

    def test_exact_boundary_1_5x_is_single(self):
        self.assertEqual(diarize.make_windows([(1.0, 5.5)], win=3.0), [(1.0, 5.5, 0)])

    def test_long_segment_split_and_covers(self):
        w = diarize.make_windows([(0.0, 10.0)], win=3.0, min_len=0.8)
        self.assertEqual(w[0][0], 0.0)
        self.assertEqual(w[-1][1], 10.0)
        for a, b in zip(w, w[1:]):
            self.assertAlmostEqual(a[1], b[0])
        self.assertTrue(all(x[2] == 0 for x in w))

    def test_short_tail_merged(self):
        w = diarize.make_windows([(0.0, 9.5)], win=3.0, min_len=0.8)
        self.assertEqual(w[-1], (6.0, 9.5, 0))
        self.assertTrue(all(e - s >= 0.8 for s, e, _ in w))

    def test_indices_follow_segments(self):
        w = diarize.make_windows([(0, 1), (2, 3)], win=3.0)
        self.assertEqual([x[2] for x in w], [0, 1])

    def test_empty(self):
        self.assertEqual(diarize.make_windows([]), [])


@unittest.skipUnless(np is not None, "numpy 없음")
class ClusterTest(unittest.TestCase):
    def _blobs(self, k, per=8, dim=16, seed=0):
        rng = np.random.RandomState(seed)
        cents = np.eye(dim)[:k] * 5.0
        emb = np.vstack([cents[i] + rng.normal(0, 0.15, (per, dim)) for i in range(k)])
        truth = np.repeat(np.arange(k), per)
        return emb, truth

    def test_trivial_sizes_without_sklearn(self):
        self.assertEqual(len(diarize.cluster_embeddings(np.zeros((0, 4)))), 0)
        self.assertEqual(diarize.cluster_embeddings(np.ones((1, 4))).tolist(), [0])

    @unittest.skipUnless(HAS_SK, "sklearn 없음")
    def test_two_and_three_speakers_auto(self):
        for k in (2, 3):
            emb, truth = self._blobs(k)
            lab = diarize.cluster_embeddings(emb)
            self.assertEqual(len(set(lab.tolist())), k)
            for i in range(k):
                self.assertEqual(len(set(lab[truth == i].tolist())), 1)

    @unittest.skipUnless(HAS_SK, "sklearn 없음")
    def test_fixed_num_speakers(self):
        emb, _ = self._blobs(3)
        self.assertEqual(len(set(diarize.cluster_embeddings(emb, num_speakers=2).tolist())), 2)
        self.assertEqual(set(diarize.cluster_embeddings(emb, num_speakers=1).tolist()), {0})
        self.assertLessEqual(len(set(diarize.cluster_embeddings(emb, num_speakers=99).tolist())), len(emb))

    @unittest.skipUnless(HAS_SK, "sklearn 없음")
    def test_max_speakers_cap(self):
        emb = np.eye(12) * 5.0
        lab = diarize.cluster_embeddings(emb, max_speakers=4)
        self.assertLessEqual(len(set(lab.tolist())), 4)


@unittest.skipUnless(np is not None, "numpy 없음")
class MergeTinyTest(unittest.TestCase):
    def test_tiny_cluster_merged_to_nearest_big(self):
        emb = np.array([[1, 0, 0], [1, 0.05, 0], [0, 1, 0], [0, 1, 0.05], [0.9, 0.1, 0]], dtype=float)
        labels = np.array([0, 0, 1, 1, 2])
        durs = np.array([30, 30, 30, 30, 1.0])
        out = diarize.merge_tiny_speakers(emb, labels, durs)
        self.assertEqual(out.tolist(), [0, 0, 1, 1, 0])
        self.assertEqual(labels.tolist(), [0, 0, 1, 1, 2])  # 입력은 바뀌지 않음

    def test_single_cluster_unchanged(self):
        emb = np.ones((3, 2))
        out = diarize.merge_tiny_speakers(emb, np.zeros(3, dtype=int), np.ones(3))
        self.assertEqual(out.tolist(), [0, 0, 0])

    def test_all_small_collapses_to_longest(self):
        emb = np.array([[1, 0], [0, 1]], dtype=float)
        out = diarize.merge_tiny_speakers(emb, np.array([0, 1]), np.array([1.0, 2.0]))
        self.assertEqual(out.tolist(), [1, 1])

    def test_big_clusters_kept(self):
        emb = np.array([[1, 0], [0, 1]], dtype=float)
        out = diarize.merge_tiny_speakers(emb, np.array([0, 1]), np.array([20.0, 20.0]))
        self.assertEqual(out.tolist(), [0, 1])


@unittest.skipUnless(np is not None, "numpy 없음")
class RelabelTest(unittest.TestCase):
    def test_first_appearance_order(self):
        names = diarize.relabel_by_first_appearance([5.0, 0.0, 9.0], np.array([3, 7, 7]))
        self.assertEqual(names, ["S2", "S1", "S1"])

    def test_three_speakers(self):
        names = diarize.relabel_by_first_appearance([0, 1, 2, 3], np.array([2, 0, 1, 2]))
        self.assertEqual(names, ["S1", "S2", "S3", "S1"])

    def test_empty(self):
        self.assertEqual(diarize.relabel_by_first_appearance([], np.array([], dtype=int)), [])


@unittest.skipUnless(np is not None, "numpy 없음")
class TurnsTest(unittest.TestCase):
    def test_merge_same_speaker_within_gap(self):
        t = diarize.segments_to_turns([(0, 2), (2.5, 4)], ["S1", "S1"], gap=1.0, max_len=28)
        self.assertEqual([(x.start, x.end, x.speaker) for x in t], [(0, 4, "S1")])

    def test_no_merge_over_gap_or_other_speaker(self):
        t = diarize.segments_to_turns([(0, 2), (4, 5), (5.2, 6)], ["S1", "S1", "S2"], gap=1.0)
        self.assertEqual([x.speaker for x in t], ["S1", "S1", "S2"])

    def test_max_len_splits(self):
        segs = [(i * 10.0, i * 10.0 + 9.5) for i in range(4)]
        t = diarize.segments_to_turns(segs, ["S1"] * 4, gap=1.0, max_len=28.0)
        self.assertEqual(len(t), 2)
        self.assertTrue(all(x.end - x.start <= 28.0 for x in t))

    def test_unsorted_input_sorted(self):
        t = diarize.segments_to_turns([(5, 6), (0, 1)], ["S2", "S1"])
        self.assertEqual([x.speaker for x in t], ["S1", "S2"])

    def test_to_dict_rounding(self):
        self.assertEqual(diarize.Turn(0.12345, 1.98765, "S1").to_dict(), {"start": 0.123, "end": 1.988, "speaker": "S1"})


@unittest.skipUnless(np is not None, "numpy 없음")
class AssignTest(unittest.TestCase):
    def test_length_weighted_vote(self):
        windows = [(0, 3, 0), (3, 4, 0), (4, 6, 1)]
        self.assertEqual(diarize.assign_segment_speakers(windows, [0, 1, 1], 2), [0, 1])

    def test_segment_without_window_defaults_zero(self):
        self.assertEqual(diarize.assign_segment_speakers([(0, 1, 0)], [2], 2), [2, 0])


if __name__ == "__main__":
    unittest.main()


class WindowTurns(unittest.TestCase):
    def test_windows_split_a_long_segment_by_speaker_and_smooth_isolated_flip(self):
        from joons_listen import diarize

        w = [(0.0, 3.0, 0), (3.0, 6.0, 0), (6.0, 9.0, 0), (9.0, 12.0, 0), (12.0, 15.0, 0)]
        turns = diarize.windows_to_turns(w, [0, 0, 1, 1, 0])
        self.assertEqual([(t.speaker, round(t.start), round(t.end)) for t in turns], [("S1", 0, 6), ("S2", 6, 12), ("S1", 12, 15)])
        # 고립된 한 창 뒤집힘은 이웃 화자로 고친다
        t2 = diarize.windows_to_turns(w, [0, 1, 0, 0, 0])
        self.assertEqual(len(t2), 1)
        self.assertEqual(t2[0].speaker, "S1")

    def test_empty(self):
        from joons_listen import diarize

        self.assertEqual(diarize.windows_to_turns([], []), [])
