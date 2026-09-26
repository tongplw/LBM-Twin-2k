"""Padded microbatches preserving equal example and equal response loss weights."""


def training_runtime(profile, mode, completed_updates):
    settings = {"microbatch": 1, "cpu_threads": 8, "checkpointing": True}
    previous = -1
    for phase in (profile or {}).get(mode, []):
        start = phase["from_update"]
        if start <= previous or start < 0:
            raise ValueError("Runtime phases must have increasing nonnegative update boundaries")
        previous = start
        if start <= completed_updates:
            settings.update({k: phase[k] for k in settings if k in phase})
    if settings["microbatch"] not in (1, 2, 4, 8, 16, 32) or settings["cpu_threads"] < 1:
        raise ValueError("Invalid runtime microbatch/thread count")
    return settings


def answer_losses(model, prepared, torch, pad_token_id, device="cuda", pad_to_width=None):
    """Return one mean answer loss per example; padding never contributes."""
    width = max(max(len(p["ids"]) for p in prepared), pad_to_width or 0)
    offsets = [width-len(p["ids"]) for p in prepared]
    inputs = torch.tensor([[pad_token_id]*offset+p["ids"] for p, offset in zip(prepared, offsets)], device=device)
    mask = torch.tensor([[0]*offset+[1]*len(p["ids"]) for p, offset in zip(prepared, offsets)], device=device)
    first = min(offset+p["spans"][0][0]-1 for p, offset in zip(prepared, offsets))
    positions = torch.arange(first, width-1, device=device)
    logits = model(input_ids=inputs, attention_mask=mask, logits_to_keep=positions, use_cache=False).logits.float()
    losses = []
    for i, (p, offset) in enumerate(zip(prepared, offsets)):
        atoms = []
        for start, end in p["spans"]:
            begin, stop = offset+start-1, offset+end-1
            atoms.append(torch.nn.functional.cross_entropy(
                logits[i, begin-first:stop-first], inputs[i, begin+1:stop+1]))
        losses.append(torch.stack(atoms).mean())
    return torch.stack(losses)
