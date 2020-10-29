import numpy as np


class SequenceDecomposer(object):
    @staticmethod
    def decompose(seq, window_width):
        assert len(seq.shape) == 2 and seq.shape[-1] == 2, "Sequence shape needs to be [sequence_length, 2]."

        seq_len = len(seq)
        if window_width == 1:
            return np.reshape(seq, [-1, 1, 2])
        if window_width >= seq_len:
            return np.reshape(seq, [1, -1, 2])

        subsequences = []
        for i in range(window_width, seq_len):
            subsequences.append(np.copy(seq[i:i+window_width]))
        return np.asarray(subsequences)





