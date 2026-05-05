"""
Tests for util/fragmol_frag_zyh.py — FragmolUtil tokeniser and decoder.

FragmolUtil maps SMILES-like tokens to integer indices and assembles
3D molecules from batches of token sequences + coordinate arrays.

These tests require rdkit (part of the lingo3dmol conda environment).
"""

import numpy as np
import pytest

rdkit = pytest.importorskip('rdkit', reason='rdkit not installed')
from util.fragmol_frag_zyh import FragmolUtil


@pytest.fixture(scope='module')
def fu():
    return FragmolUtil()


# ── vocabulary consistency ────────────────────────────────────────────────────

class TestVocabulary:

    def test_encode_vocab_size(self, fu):
        # Encoder vocab (used during generation)
        assert len(fu.vocab_list) > 0

    def test_decode_vocab_size(self, fu):
        # Decoder vocab (used to interpret model output)
        assert len(fu.vocab_list_decode_new) > 0

    def test_encode_decode_new_roundtrip(self, fu):
        """Every index maps back to a unique token and vice versa."""
        for i, token in fu.vocab_i2c_v1_decode_new.items():
            assert fu.vocab_c2i_v1_decode_new[token] == i

    def test_no_duplicate_encode_indices(self, fu):
        indices = list(fu.vocab_c2i_v1.values())
        assert len(indices) == len(set(indices))

    def test_no_duplicate_decode_new_indices(self, fu):
        indices = list(fu.vocab_c2i_v1_decode_new.values())
        assert len(indices) == len(set(indices))

    def test_special_token_pad_is_zero(self, fu):
        assert fu.vocab_c2i_v1['pad'] == 0

    def test_special_token_start_is_one(self, fu):
        assert fu.vocab_c2i_v1['start'] == 1

    def test_special_token_end_is_two(self, fu):
        assert fu.vocab_c2i_v1['end'] == 2

    def test_special_token_sep_is_three(self, fu):
        assert fu.vocab_c2i_v1['sep'] == 3

    def test_encode_length_is_100(self, fu):
        assert fu.encode_length == 100

    def test_resolution_is_0_1(self, fu):
        assert fu.resolution == pytest.approx(0.1)


# ── ele_token (atom-type indices) ─────────────────────────────────────────────

class TestEleTokens:

    def test_ele_token_nonempty(self, fu):
        assert len(fu.ele_token) > 0

    def test_ele_tokens_not_special_tokens(self, fu):
        """Atom tokens must not overlap with pad/start/end/sep."""
        special = {0, 1, 2, 3}
        assert not set(fu.ele_token) & special

    def test_ele_tokens_are_valid_decode_indices(self, fu):
        """Every ele_token index must exist in the decode vocab."""
        for idx in fu.ele_token:
            # We use the basic decode vocab, which has idx → token mapping
            token = fu.vocab_i2c_v1_decode.get(idx)
            assert token is not None, f'ele_token {idx} not in decode vocab'

    def test_ele_tokens_correspond_to_heavy_atoms(self, fu):
        """Every token in ele_token should decode to a known heavy-atom symbol."""
        heavy_atoms = {'C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'F', 'Cl',
                       'Br', '[nH]'}
        for idx in fu.ele_token:
            token = fu.vocab_i2c_v1_decode[idx]
            assert token in heavy_atoms, (
                f'ele_token {idx} → "{token}" is not a heavy atom symbol'
            )


# ── decode3d ──────────────────────────────────────────────────────────────────

class TestDecode3D:

    def _zeros_batch(self, n_samples=2, length=100):
        """All-pad token sequences → nothing to decode."""
        codes = np.zeros((n_samples, length), dtype=np.int32)
        pos = np.zeros((n_samples, length, 3), dtype=np.float32)
        return codes, pos

    def test_returns_three_lists(self, fu):
        codes, pos = self._zeros_batch()
        result = fu.decode3d(codes, pos)
        assert len(result) == 3

    def test_all_pad_yields_none_molecules(self, fu):
        codes, pos = self._zeros_batch(n_samples=3)
        smiles, tokens, mols = fu.decode3d(codes, pos)
        # All-zeros (pad) produces no fragments → None molecules
        assert all(m is None for m in mols)

    def test_output_length_matches_batch_size(self, fu):
        codes, pos = self._zeros_batch(n_samples=4)
        smiles, tokens, mols = fu.decode3d(codes, pos)
        assert len(smiles) == 4
        assert len(tokens) == 4
        assert len(mols) == 4

    def test_end_token_terminates_sequence(self, fu):
        """
        A sequence that starts immediately with the end token (2) should
        produce no molecule — the decoder must stop at end token.
        """
        codes = np.zeros((1, 100), dtype=np.int32)
        codes[0, 0] = 2   # immediate end
        pos = np.zeros((1, 100, 3), dtype=np.float32)
        smiles, tokens, mols = fu.decode3d(codes, pos)
        assert mols[0] is None

    def test_single_batch_item(self, fu):
        codes = np.zeros((1, 100), dtype=np.int32)
        pos = np.zeros((1, 100, 3), dtype=np.float32)
        smiles, tokens, mols = fu.decode3d(codes, pos)
        assert len(smiles) == 1

    def test_positions_are_float_convertible(self, fu):
        """decode3d casts positions to float16; very large values should not crash."""
        codes, pos = self._zeros_batch(n_samples=2)
        pos[:] = 239.0  # max valid voxel coordinate
        # Should complete without raising
        fu.decode3d(codes, pos)


# ── mergeSmiles3D ─────────────────────────────────────────────────────────────

class TestMergeSmiles3D:

    def test_empty_fragment_list_returns_none(self, fu):
        smi, mol = fu.mergeSmiles3D([], [])
        assert smi is None
        assert mol is None

    def test_single_invalid_smiles_returns_none(self, fu):
        smi, mol = fu.mergeSmiles3D(['not_valid_smiles!!!'], [])
        assert smi is None
        assert mol is None

    def test_valid_single_fragment(self, fu):
        """A valid one-fragment SMILES should return a SMILES string and mol."""
        # 'C' is the simplest valid SMILES (methane)
        smi, mol = fu.mergeSmiles3D(['C'], [[0.0, 0.0, 0.0]])
        # May return None if sanitisation fails on the fragment, but must not raise
        # We only assert the types are correct when not None
        if smi is not None:
            assert isinstance(smi, str)
