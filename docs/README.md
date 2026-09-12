# `docs/` — what goes here

Two things live in this folder, and **both are validity gates**. Neither earns you marks by
being present. They are priced differently, so the table is the authority:

| Path | What it is | If missing |
|---|---|---|
| `docs/cycle-transcript.md` | Your Cycle session with ChatGPT — an export, or a file containing the share link | **−6 raw stage points** |
| `docs/meetings/` | Every team meeting's transcript **and** its minutes, as separate files — **or** a `README.md` here declaring the restricted-Moodle route | Condition of assessment: your individual contribution mark loses one of its three sources and cannot be awarded above 4 of 6 (rubric §6) |

## `docs/cycle-transcript.md`

The whole session, not a summary. The Cycle stage is marked on whether you **challenged** the
model's mechanisms rather than accepting its first answer, and the transcript is the only
evidence of that. A reconstructed-after-the-fact transcript is visible as one and scores zero.

**The file must exist here**, as `docs/cycle-transcript.md` or `docs/cycle-transcript.pdf` —
the completeness check looks for it by name. Export the session from ChatGPT into it, or put
the share link in it; either is accepted, but a link that lives only in your report does not
satisfy the gate.

## `docs/meetings/`

**Separate files per meeting, named so the check can pair them.** For a meeting on
16 March 2026:

```
docs/meetings/2026-03-16-transcript.md
docs/meetings/2026-03-16-minutes.md
```

`.md`, `.txt`, `.pdf` or `.docx`; a per-date subfolder such as
`docs/meetings/2026-03-16/transcript.md` also works. **One combined file per meeting does
not pass** — the check requires both halves for every date it finds, because the transcript
is evidence the meeting happened and the minutes are the part that is marked. At least two
dated meetings are required, each fully paired.

The minutes show what the team **decided**, who owns each action, and what is still open.
They are one of the three sources your **individual** contribution mark comes from —
alongside the repository history and both peer-assessment rounds — so a meeting record that
never names who argued for what cannot help you.

### Or the restricted-Moodle route

Identifiable records may go to the restricted Moodle item instead. Declare it here, on its
own line in `docs/meetings/README.md`:

```
SUBMISSION ROUTE: <Moodle or repository>
```

with the real word in place of the placeholder.

Then this folder holds only that README. Mentioning Moodle anywhere else in the file changes
nothing — if records are present here, they are checked here.

See `team-templates/meeting-records-requirements.md` on Moodle for the template, the worked
example, the route declaration and how recordings are handled.
