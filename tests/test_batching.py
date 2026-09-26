import pytest

from twinlab.batching import training_runtime, answer_losses


def test_runtime_switch_preserves_original_completed_updates():
    profile = {"question_only": [{"from_update": 32, "microbatch": 8, "checkpointing": False}]}
    assert training_runtime(profile, "question_only", 31) == training_runtime(None, "question_only", 0)
    assert training_runtime(profile, "question_only", 32)["microbatch"] == 8
    assert training_runtime(profile, "full_history", 100)["microbatch"] == 1


def test_runtime_rejects_invalid_or_unordered_phases():
    with pytest.raises(ValueError):
        training_runtime({"question_only": [{"from_update": 32}, {"from_update": 20}]}, "question_only", 50)
    with pytest.raises(ValueError):
        training_runtime({"question_only": [{"from_update": 0, "microbatch": 7}]}, "question_only", 0)


def test_padded_batch_matches_individual_answer_weighting_and_gradients():
    from types import SimpleNamespace
    import torch
    torch.manual_seed(42)

    class TinyCausalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(12, 4)
            self.head = torch.nn.Linear(4, 12)

        def forward(self, input_ids, attention_mask, logits_to_keep, use_cache):
            hidden = (self.embedding(input_ids)*attention_mask[..., None]).cumsum(1)
            return SimpleNamespace(logits=self.head(hidden[:, logits_to_keep]))

    model = TinyCausalModel()
    # Different sequence lengths, answer lengths and numbers of response rows.
    examples = [{"ids": [1, 2, 3], "spans": [(2, 3)]},
                {"ids": [4, 5, 6, 7, 8, 9], "spans": [(3, 5), (5, 6)]}]
    single = torch.stack([answer_losses(model, [e], torch, 0, "cpu")[0] for e in examples])
    single.mean().backward()
    gradients = {k: p.grad.clone() for k, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    batched = answer_losses(model, examples, torch, 0, "cpu")
    assert torch.allclose(single, batched, atol=1e-6)
    batched.mean().backward()
    assert all(torch.allclose(p.grad, gradients[k], atol=1e-6) for k, p in model.named_parameters())
