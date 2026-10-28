# Sort with Evidence
A local, review-first campus inbox classifier using actual multinomial Naive Bayes, alpha=1 Laplace smoothing. Live demo: https://adityam274.github.io/SortWithEvidence/ .

Run `python -m http.server 4182 --bind 127.0.0.1`; open localhost4182. Test: `node test.mjs`. No install or API key.

## Workflow
Classify the six-message fixture. Inspect score, vocabulary coverage and per-word evidence. Select a final label (or Unassigned), then approve each decision. Export includes all inputs, predictions, final labels, approvals, training examples and correction provenance. Approved corrections can train a new model; prior review is invalidated. Nothing sends/routes a message. Data is in page memory only; reload clears it. External Google Fonts load typography but receive no message payload.

## Validation and limits
24original synthetic training examples,8separate synthetic held-out cases:8/8suggested labels correct on this narrow fixture. Not real-world accuracy. Unknown vocabulary and ambiguous text require review; scores are not calibrated probabilities. Tests include an independent hand calculation, overlap/duplicate-ID rejection, bounds and score normalization. Browser six-item review and actual downloaded audit, invalid input, approved correction and review invalidation verified. Narrow390x844iframe layout/core classification checked; not a physical phone. No user study, field deployment or time savings claimed.

## Attribution
OpenAI Codex generated all new code, fixtures and documentation on28October2026 under Aditya Mittal’s direction. Runtime uses a custom implementation of the standard multinomial NB algorithm, no LLM. No previous competition code reused. Algorithm reference and limitations: https://scikit-learn.org/stable/modules/naive_bayes.html . Original synthetic text is not real correspondence.

Future: collect consented labels, measure drift/calibration against a real held-out set, support more languages and explicit routing policies. Automatic sending is outside current scope.
