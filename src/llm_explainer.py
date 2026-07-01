"""
OPTIONAL: natural-language explanations for recommendations.

The core recommendation engine needs NO API key and runs fully offline. This
module is a bonus "advanced twist": it asks a large language model to turn the
numeric recommendation signals into a friendly, human sentence -- the kind of
"Because you watched ..." blurb real e-commerce sites show.

Supports either Anthropic (Claude) or OpenAI. It reads the key from an
environment variable and degrades gracefully (returns a templated explanation)
if no key or no SDK is available, so the rest of the project never breaks.

Set ONE of these in your environment (see README):
    ANTHROPIC_API_KEY=sk-ant-...
    OPENAI_API_KEY=sk-...
"""
from __future__ import annotations

import os
import textwrap

from .data_loader import Dataset


def _liked_titles(dataset: Dataset, user_id: int, n: int = 5) -> list[str]:
    df = dataset.train
    liked = (
        df[(df.userId == user_id) & (df.rating >= 4.0)]
        .sort_values("rating", ascending=False)
        .head(n)
    )
    return [dataset.title(m) for m in liked.movieId]


def _template_explanation(liked: list[str], recommended_title: str) -> str:
    if liked:
        basis = ", ".join(liked[:3])
        return (
            f"Recommended '{recommended_title}' because you rated "
            f"{basis} highly -- it shares themes and audience with those."
        )
    return f"Recommended '{recommended_title}' based on overall popularity and fit."


def explain_recommendation(
    dataset: Dataset,
    user_id: int,
    recommended_movie_id: int,
    provider: str | None = None,
) -> str:
    """Return a one- or two-sentence explanation. Uses an LLM if a key exists."""
    liked = _liked_titles(dataset, user_id)
    recommended_title = dataset.title(recommended_movie_id)

    provider = provider or ("anthropic" if os.getenv("ANTHROPIC_API_KEY")
                            else "openai" if os.getenv("OPENAI_API_KEY") else None)

    prompt = textwrap.dedent(f"""
        A user enjoyed these movies: {liked}.
        Our recommender suggests: "{recommended_title}".
        In ONE friendly sentence (max 30 words), explain why they might like it.
        Start with "Because you enjoyed".
    """).strip()

    try:
        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()

        if provider == "openai":
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
    except Exception as exc:  # missing SDK, bad key, network, etc.
        return _template_explanation(liked, recommended_title) + f"  (LLM unavailable: {exc.__class__.__name__})"

    return _template_explanation(liked, recommended_title)


if __name__ == "__main__":
    from .data_loader import load_dataset
    from .hybrid import HybridRecommender

    ds = load_dataset()
    hybrid = HybridRecommender(ds).fit()
    uid = 1
    top_mid, _ = hybrid.recommend(uid, top_n=1)[0]
    print(explain_recommendation(ds, uid, top_mid))
