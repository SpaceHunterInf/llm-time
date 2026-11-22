"""Offline typesetting for dashboard equations and notation in historical prose."""

from __future__ import annotations

from html import escape
import re


def equation(markup: str, plain: str) -> str:
    """Keep a readable text equivalent for viewers without native MathML layout."""
    return (
        '<div class="formula-line" tabindex="0" role="group" aria-label="Equation">'
        f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="{escape(plain)}">'
        f'<mrow>{markup}</mrow></math><span class="formula-text">{escape(plain)}</span></div>'
    )


def score_equation() -> str:
    return equation(
        '<mi>U</mi><mo>=</mo><mn>100</mn><mo>×</mo><mfrac>'
        '<mrow><mo>∑</mo><msub><mi>n</mi><mi>t</mi></msub></mrow>'
        '<mrow><mo>∑</mo><msubsup><mi>n</mi><mi>t</mi><mo>⋆</mo></msubsup></mrow></mfrac>',
        "U = 100 × (sum of the agent’s expected net scores) / (sum of the reference’s net scores)",
    )


def historical_math_cards() -> str:
    task = equation(
        '<msub><mi>M</mi><mi>t</mi></msub><mo>=</mo><mo>(</mo><mi>S</mi><mo>,</mo>'
        '<msub><mi>A</mi><mtext>comp</mtext></msub><mo>,</mo><mi>T</mi><mo>,</mo>'
        '<msub><mi>R</mi><mi>t</mi></msub><mo>,</mo><mi>Ω</mi><mo>)</mo>',
        "The task at time t consists of states, compliant actions, transition dynamics, rewards and observations.",
    )
    shift = equation(
        '<msub><mi>R</mi><mi>t</mi></msub><mo>=</mo>'
        '<mi>R</mi><mo>(</mo><msub><mi>s</mi><mi>t</mi></msub>'
        '<mo>,</mo><msub><mi>a</mi><mi>t</mi></msub><mo>,</mo><msub><mi>z</mi><mi>t</mi></msub><mo>)</mo>',
        "Reward at time t depends on the state, action and hidden payoff regime.",
    )
    session_utility = equation(
        '<msub><mi>U</mi><mi>t</mi></msub><mo>=</mo><mfrac>'
        '<mrow><mi>max</mi><mo>(</mo><mn>0</mn><mo>,</mo><msub><mi>g</mi><mi>t</mi></msub>'
        '<mo>−</mo><msub><mi>c</mi><mi>t</mi></msub><mo>)</mo></mrow>'
        '<mrow><mi>max</mi><mo>(</mo><mn>1</mn><mo>,</mo>'
        '<msubsup><mi>g</mi><mi>t</mi><mo>⋆</mo></msubsup><mo>)</mo></mrow></mfrac>',
        "Session utility = max(0, expected portfolio yield − fees) divided by max(1, best compliant expected yield).",
    )
    paired_gain = equation(
        '<msubsup><mi>Δ</mi><mi>w</mi><mi>s</mi></msubsup><mo>=</mo><mfrac><mn>1</mn><mi>H</mi></mfrac>'
        '<munderover><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>H</mi></munderover>'
        '<mo>(</mo><msub><mi>U</mi><mrow><mi>A</mi><mo>,</mo><mi>j</mi></mrow></msub>'
        '<mo>−</mo><msub><mi>U</mi><mrow><mi>B</mi><mo>,</mo><mi>j</mi></mrow></msub><mo>)</mo>',
        "For seed s in world w, the paired gain is the mean difference in session utility between arms A and B after the checkpoint.",
    )
    difference = equation(
        '<msub><mi>Δ</mi><mtext>adjusted</mtext></msub><mo>=</mo>'
        '<msub><mi>Δ</mi><mtext>change</mtext></msub><mo>−</mo><msub><mi>Δ</mi><mtext>stable</mtext></msub>',
        "Adjusted gain = gain in the changing world − gain in the matched stable world.",
    )
    surprise = equation(
        '<msub><mi>S</mi><mi>t</mi></msub><mo>=</mo><mi>max</mi><mo>(</mo><mn>0</mn><mo>,</mo>'
        '<msub><mi>S</mi><mrow><mi>t</mi><mo>−</mo><mn>1</mn></mrow></msub><mo>+</mo>'
        '<mover><mi>μ</mi><mo>^</mo></mover><mo>−</mo><msub><mi>y</mi><mi>t</mi></msub>'
        '<mo>−</mo><mi>δ</mi><mo>)</mo>',
        "Accumulated surprise = max(0, previous surprise + expected yield − observed yield − slack).",
    )
    slope = equation(
        '<msub><mover><mi>r</mi><mo>¯</mo></mover><mi>t</mi></msub><mo>=</mo><mo>(</mo><mn>1</mn>'
        '<mo>−</mo><mi>β</mi><mo>)</mo><msub><mover><mi>r</mi><mo>¯</mo></mover>'
        '<mrow><mi>t</mi><mo>−</mo><mn>1</mn></mrow></msub><mo>+</mo><mi>β</mi><mo>(</mo>'
        '<msub><mi>y</mi><mi>t</mi></msub><mo>−</mo><mover><mi>μ</mi><mo>^</mo></mover><mo>)</mo>',
        "Smoothed residual = (1 − β) × previous residual + β × (observed − expected yield).",
    )
    brier = equation(
        '<msub><mi>B</mi><mi>t</mi></msub><mo>=</mo><msup><mrow><mo>(</mo>'
        '<msub><mi>p</mi><mi>t</mi></msub><mo>−</mo><msub><mi>o</mi><mi>t</mi></msub>'
        '<mo>)</mo></mrow><mn>2</mn></msup>',
        "Brier loss = (predicted probability − observed binary outcome) squared.",
    )
    training = equation(
        '<mi>J</mi><mo>=</mo><munder><mo>∑</mo><mi>t</mi></munder><mo>(</mo>'
        '<msub><mi>U</mi><mi>t</mi></msub><mo>−</mo><mi>η</mi>'
        '<msub><mi>B</mi><mi>t</mi></msub><mo>)</mo>',
        "Proposed objective = sum of net utility minus a weighted Brier loss across sessions.",
    )
    return f"""
    <article class="bento-card bento-6"><h3 class="card-title">How the benchmark is scored</h3>
      <div class="math-panel"><h4>The partially observed task</h4>{task}{shift}
      <p><var>S</var> is the state space, <var>A</var><sub>comp</sub> the legal allocations, <var>T</var> the transition rule and <var>Ω</var> the observations available to the agent. The hidden regime <var>z</var><sub>t</sub> changes the reward for taking the same action in the same observed conditions.</p></div>
      <div class="math-panel"><h4>AlphaDesk v0.2 · one session</h4>{session_utility}
      <p>For a valid allocation, <var>g</var><sub>t</sub> is its expected gross yield. The fee <var>c</var><sub>t</sub> includes 12 points for a probe and 18 if the free desk brief was skipped. The reference chooses the best legal weights, which sum to one with at most 65% per strategy. A ratio of one is reported as 100%.</p></div>
      <div class="math-panel"><h4>Shared-checkpoint comparison in v0.2</h4>{paired_gain}{difference}
      <p><var>j</var> indexes the <var>H</var> sessions after the same starting checkpoint for both arms. The early study used 16 evaluation sessions and a seeded change point between 12 and 16. Compute the gap for each seed in both worlds, then subtract its stable-world gap and average across seeds.</p></div>
      <div class="math-panel"><h4>AlphaDesk v0.3 · a sequence of decisions</h4><p>A probe costs 10 points, the weight cap is 55% and moving between holdings incurs a fee. The reference also pays switching costs, so this version compares cumulative net scores over the same horizon. <a href="#score-guide">See the v0.3 scoring equation</a>.</p></div>
    </article>
    <article class="bento-card bento-6"><h3 class="card-title">Checking rules and proposed training</h3>
      <div class="math-panel"><h4>Detecting a shortfall</h4>{surprise}{slope}
      <p>Within the same market conditions, the controller accumulates shortfalls beyond a tolerance δ. A smoothing weight β sets how quickly the average responds. A separate clock tracks how long options have gone unchecked.</p></div>
      <div class="math-panel"><h4>Forecast calibration · deferred proposal</h4>{brier}{training}
      <p>The proposal asks for the chance <var>p</var> that a fixed reference action clears 75 points. Its outcome <var>o</var> is one or zero. Smaller forecast errors would increase the objective, with weight η. The reported experiments used frozen models.</p></div>
    </article>"""


# Literal transcriptions used in the older handbook. Apply only to prose.
# Prompts, JSON, command examples and trace records keep their original bytes.
_SYMBOLS = {
    "mu_hat_a": "μ̂<sub>a</sub>", "mu_hat": "μ̂", "mu_pre": "μ<sub>pre</sub>",
    "mu_rev": "μ<sub>rev</sub>", "mu_t": "μ<sub>t</sub>", "lambda_t": "λ<sub>t</sub>",
    "H_post": "H<sub>post</sub>", "H_haz": "H<sub>haz</sub>", "w_{t-1}": "w<sub>t−1</sub>",
    "w_t": "w<sub>t</sub>", "w_i": "w<sub>i</sub>", "y_t": "y<sub>t</sub>",
    "U_t": "U<sub>t</sub>", "sum_t": "∑<sub>t</sub>", "R_{Brier,t}": "R<sub>Brier,t</sub>",
    "eta": "η", "Delta t_a": "Δt<sub>a</sub>", "M_{t*}": "M<sub>t⋆</sub>",
    "t*": "t<sup>⋆</sup>", "||_1": "‖<sub>1</sub>", "||": "‖",
    " -> ": " → ", " >= ": " ≥ ", " <= ": " ≤ ", " * ": " × ",
}
_TOKENS = re.compile("|".join(re.escape(k) for k in sorted(_SYMBOLS, key=len, reverse=True)))
_CODE = re.compile(r"(`[^`\n]+`)")


def prose(value: object) -> str:
    """Escape text first conceptually: only known notation can introduce markup."""
    parts = []
    for piece in _CODE.split(str(value or "")):
        is_code = piece.startswith("`") and piece.endswith("`") and len(piece) > 1
        raw = piece[1:-1] if is_code else piece
        if is_code and not any(token in raw for token in ("w_i", "w_t", "w_{", "mu_hat", "t*", "Delta t_a", "H_post")):
            parts.append(f"<code>{escape(raw)}</code>")
            continue
        start = 0
        for match in _TOKENS.finditer(raw):
            # Short Greek transliterations can also occur inside ordinary words.
            if match[0] == "eta" and ((match.start() and raw[match.start()-1].isalpha())
                                      or (match.end() < len(raw) and raw[match.end()].isalpha())):
                continue
            parts.extend((escape(raw[start:match.start()]), _SYMBOLS[match[0]]))
            start = match.end()
        parts.append(escape(raw[start:]))
    return "".join(parts)
