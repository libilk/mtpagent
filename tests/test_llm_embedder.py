import pytest
from unittest.mock import MagicMock

from llm.embedder import (
    Embedder,
    cosine_similarity,
    compute_similarities,
    rank_by_similarity,
    VectorMatcher,
)


class TestEmbedder:
    def test_embed_single_text(self, mock_embedder):
        emb = Embedder(api_embedder=mock_embedder)
        result = emb.embed("hello")
        assert result == [0.1, 0.2, 0.3]
        mock_embedder.embed_query.assert_called_once()
        assert mock_embedder.embed_query.call_args[0][0] == "hello"

    def test_embed_no_embedder_raises(self):
        emb = Embedder(api_embedder=None)
        with pytest.raises(ValueError, match="未配置嵌入器"):
            emb.embed("hello")

    def test_embed_batch(self, mock_embedder):
        emb = Embedder(api_embedder=mock_embedder)
        result = emb.embed_batch(["a", "b"])
        assert len(result) == 2
        mock_embedder.embed_documents.assert_called_once_with(["a", "b"])

    def test_embed_batch_no_embedder_raises(self):
        emb = Embedder(api_embedder=None)
        with pytest.raises(ValueError, match="未配置嵌入器"):
            emb.embed_batch(["a", "b"])


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_negative_correlation(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


class TestComputeSimilarities:
    def test_returns_list(self):
        sims = compute_similarities([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
        assert len(sims) == 2
        assert sims[0] == pytest.approx(1.0)
        assert sims[1] == pytest.approx(0.0)


class TestRankBySimilarity:
    def test_rank_descending(self):
        query = [1.0, 0.0]
        candidates = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
        vecs = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
        ranked = rank_by_similarity(query, candidates, vecs, top_k=3)
        assert ranked[0][0]["name"] == "B"  # highest similarity
        assert ranked[2][0]["name"] == "A"  # lowest similarity

    def test_top_k_truncation(self):
        query = [1.0, 0.0]
        candidates = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
        vecs = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
        ranked = rank_by_similarity(query, candidates, vecs, top_k=2)
        assert len(ranked) == 2


class TestVectorMatcher:
    def test_add_to_index(self, mock_embedder):
        vm = VectorMatcher(embedder=mock_embedder)
        vm.add("id1", {"name": "test"}, "some text")
        assert "id1" in vm.index
        assert vm.index["id1"]["data"] == {"name": "test"}

    def test_search_returns_best_match(self, mock_embedder):
        mock_embedder.embed.side_effect = [
            [1.0, 0.0],  # for "a"
            [0.0, 1.0],  # for "b"
            [1.0, 0.0],  # query
        ]
        vm = VectorMatcher(embedder=mock_embedder)
        vm.add("a", {"name": "A"}, "text a")
        vm.add("b", {"name": "B"}, "text b")
        results = vm.search("query", top_k=1)
        assert len(results) == 1
        assert results[0][0]["name"] == "A"

    def test_search_empty_index(self, mock_embedder):
        vm = VectorMatcher(embedder=mock_embedder)
        assert vm.search("anything") == []

    def test_clear(self, mock_embedder):
        vm = VectorMatcher(embedder=mock_embedder)
        vm.add("id1", {}, "text")
        vm.clear()
        assert len(vm.index) == 0
