# Interface contracts

One file per hand-over between stages. **Agree and commit these in Week 2, before writing the code
they describe.**

A contract states exactly what one stage passes to the next: the file, the columns, the types, what
each column means, and what a missing value means. It must be clear enough that the receiving
member can start work before the sending member has finished.

`stage3_to_stage4.md` is a completed worked example. Copy its format.

**Contracts are tested.** `tests/test_contracts.py` reads these files and checks real stage output
against them. A prose contract gets violated silently; a tested one fails at the boundary and names
the stage that broke it.

When a contract changes, **both owners** update this file in the same pull request.
