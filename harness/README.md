# Supplied harness — do not modify

Tools for section 8.3 of the brief. **You run these and interpret the results.** Rebuilding them
earns no marks; interpreting them earns most of the 15.

Both take *your* functions as arguments, so they work with whatever you build in stage 3.

## perturbation.py

Changes specific wording in a document, re-scores it, and reports how far the score moved.

```python
from harness.perturbation import perturbation_sweep, DEFAULT_EDITS
# TRAP 5: a single scoring call is as noisy as the effect you are measuring.
# score_field() averages n calls - use it, and pass the RETRIEVED passage, not the whole document.
from src.stage3_riskvoice import score_field
my_scorer = lambda t: score_field(t, "financial_conditions_concern", n=5)

res = perturbation_sweep(score_fn=my_scorer, text=doc, edits=DEFAULT_EDITS)
```

## faithfulness.py

Asks the model why it scored a document as it did, then tests whether the reason it gave actually
drives its behaviour.

```python
from harness.faithfulness import faithfulness_test
fx = faithfulness_test(score_fn=my_scorer, explain_fn=my_explainer, text=doc)
print(fx["faithfulness_gap"])
```

A large gap means the model's explanation sounds convincing but does not describe what it is
actually doing. **This is a finding, not a bug.** Report it.
