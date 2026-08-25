# TRAPS

Nine issues that cost hours and teach nothing. **Three are already fixed in the code you are
given. Six are live in the code and the judgements you supply.**

---

## Already fixed — do not re-break them

### 1. The hold observations do not exist in any source file

RBA table A2 lists only the meetings that **moved**. The 156 holds are created in
`data_panel.attach_decisions()` by joining A2 onto the meeting calendar and defaulting the
remainder to zero. It is done once, explicitly, and it warns if any announced change fails
to match a meeting.

**If you rebuild this join yourself with a plain merge, a failed join and a genuine hold
look identical** — and you will not find out.

### 2. Publication lags

Every macro series is stamped with the date it became **known**, not the period it
describes, and joined backward as-of in `data_macro.as_of()`. At a typical meeting the
newest CPI print is 64 days old and GDP is 122 days old.

Changing `direction="backward"` to `"nearest"` or `"forward"` will improve every number in
your report and invalidate all of them.

### 3. Text from the meeting you are predicting — FIXED FOR YOU

Minutes for meeting *T* publish about 14 days after *T*, **and they state the decision taken
at T**. Scoring them in full is fine — both targets look forward and the panel carries
`decision` as a number anyway — but using a meeting's own minutes to say something about that
same meeting is not.

The fix is a single lag: `data_panel` shifts the **whole text tier by one meeting**, so the
construct values on row *M* come from the minutes of *M−1*. Those published about a fortnight
after *M−1*, and meetings are roughly five weeks apart, so they were public before *M*.
`data_panel._assert_text_available()` fails the build if any meeting gap is short enough to
break that.

Because the lag already does the work, `decision_replay.load_as_at()` **keeps** the construct
values on the target row and blanks only the decision, the rate change and the targets. An
earlier version blanked the constructs too, which sounded cautious and was actually a bug: it
showed the model seven construct values for every historical example and none for the meeting
it was being asked to judge.

---

## Live in the code and the judgements you supply

### 4. Rolling origin, or nothing

`sklearn.model_selection.cross_val_score` and `train_test_split` both leak here. Policy is
strongly autocorrelated, so a randomly chosen test meeting usually sits between two training
meetings that between them nearly give the answer away. Your numbers will look excellent.

Use `evaluation.rolling_origin()`. **This is the rubric's −20 mark validity gate.**

### 5. The scaler must be refitted inside the window

`StandardScaler().fit(X)` before the loop leaks the future's mean and variance into every
training set. `evaluation.default_classifier()` returns a `Pipeline`, and `rolling_origin()`
clones and refits it at each step. If you write your own classifier, wrap the scaler in the
pipeline — do not scale up front.

### 6. N-shot examples must predate the target — in three different senses

`nshot` enforces all three, and each raises rather than warning.

**Dates.** `select_shots()` rejects any example dated on or after your Replay meeting.

**Labels.** The `regimes` strategy selects on `y_cycle`, which is defined by what the cash
rate does over the *following* 182 days. A meeting two months before your target has a
`y_cycle` that nobody could have known at your target. `_label_known_by()` drops those.
Without it the strategy quietly selects on the future and looks unbeatable.

**The question block.** `build_prompt_block()` refuses a target row that still carries a
decision, so you cannot build the question from the raw panel by accident. Pass
`as_at=load_as_at(meeting)`.

### 7. Do not choose your strategy on the meetings you then report

Four strategies, and the best of four always looks better than it is. Iterate on
`dev_sample()`; run `holdout_sample()` **once** and report what it gives you. Our own worked
solution found all four strategies tied at 0.778 on dev — a four-way tie is a result, and
reporting the "winner" would have been reporting a coin toss.

### 8. Never shock a policy-outcome variable

The cash rate, the decision and the trailing rate-change measures **are** the policy stance. A
Shock channel whose proxy is `cash_rate` says "the RBA tightens, therefore the model predicts
tightening" — it asserts its own conclusion and the model faithfully agrees.
`scenario_engine` refuses them and marks the channel unmodellable, keeping its reasoning as
text. Shock the *economy*; let the model say what policy does about it.

### 9. A probability from outside the model's support is not a forecast

A gradient-boosted tree splits on thresholds it saw in training. Push an input past the
observed range and every split has already fired: the prediction freezes at the edge value
and stops responding, however much further you push. The number that comes back is confident
and computed from nothing.

`support_check()` tests every shocked variable against the rows the model was actually
FITTED ON, and adds a joint nearest-neighbour check. When either fails, `run_scenario()`
returns `model_verdict = "out_of_support"` with **no probability AND no direction**. The
argmax of an extrapolated distribution is the same claim as the distribution, so suppressing
one and publishing the other is not a safeguard.

What remains sayable is `channel_direction()`: a tally of YOUR adjudicated channels,
aggregated by mechanism, carrying no probability and labelled as human reasoning wherever it
appears. Both worked scenarios land out of support, and "the model cannot speak here,
here is why, and here is what our channels say instead" is the complete answer. Quoting a
model probability or a model direction anyway is an 8-mark gate.
