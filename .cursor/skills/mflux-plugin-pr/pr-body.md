# Writing the PR body

The body is drafted in a file at the repository root, shown to the developer, approved, read back, and only then
sent to GitHub. Both draft files are ignored by git (see the root `.gitignore`).

## 1. Collect the facts

Use these sources and nothing else:

- `git log --oneline origin/main..HEAD` for the commits (after `git fetch origin`, so the base is current).
- `git diff --stat origin/main...HEAD` for the changed files, and `git diff origin/main...HEAD` for the change.
- For a PR that is already open: `gh pr view <n> --json title,body` and `gh pr checks <n>`.
- The checks you ran in the skill's step 3, each with what it printed.

## 2. Create the draft

If `tmp-PR-content.md` already exists, ask the developer before you replace it. Otherwise create it with:

- line 1: `# Draft PR Content`
- line 2: `Title: ` followed by the title
- line 3: empty
- then the whole of `.github/pull_request_template.md`.

The title is one short line that names the change. For an open PR, keep its current title unless the developer asks
for a new one. `gh pr edit` replaces the whole body, so copy into the draft any section or link of the open PR's body
that the template lacks, or ask the developer before dropping it.

## 3. Fill it in

- **What.** Replace the comment with one paragraph: what changes, why, and who notices. Use the words a user of the
  plugin knows; explain a term the first time it appears. Write only about what this PR changes, with numbers only
  from the diff or a committed benchmark, and no praise words. Add `Fixes #<n>` for each issue the PR closes.
  - Plugin PR: name the plugin folder, and the `template/` change if one rides along.
  - Repo-level PR: name the root files that changed, for example `.github/workflows/lint.yml` or `CONTRIBUTING.md`.
- **Checklist.** Tick a box only when a command you ran shows it holds. Leave every other box unticked, and never
  write a note next to a box.
  - Plugin PR: "One plugin per pull request" when the diff stat shows one plugin folder; the tests box when the
    plugin's tests, ruff and ty passed; the checker box when `scripts/check_plugins.py` passed; the `README.md` and
    `CHANGELOG.md` box when the diff holds every update the change needs (a docs-only or test-only change needs no
    CHANGELOG line, so its `README.md` update alone is enough).
  - Repo-level PR: tick the checker box when both `scripts/check_plugins.py` and `scripts/check_skills.py` passed.
    The plugin-only boxes stay unticked: "One plugin per pull request", the plugin's tests, ruff and ty, its
    `README.md` and `CHANGELOG.md`, and the two new-plugin lines.
  - The drift box and "It meets the acceptance criteria" state the developer's own judgment: leave them for the
    developer to tick.
  - Any `tests/test_mflux_drift.py` digest changed, in a plugin or in `template/`, on a plugin or a repo-level PR:
    on the drift box's line, replace `<sha>` with the commit from that file's "Recorded against mflux …" comment.
    The comment must have changed in this diff together with the digest
    (`git diff origin/main...HEAD -- <folder>/tests/test_mflux_drift.py`). If it did not, ask the developer which
    mflux commit they read. Never work the sha out yourself.
  - New plugin: the proposal issue number goes after `#`.
- **Verification.** Replace the comment with the commands that ran, each followed by what it printed (the summary
  lines as printed, in a fenced block). A red result is quoted as printed, followed by one sentence saying the
  developer chose to open the PR with it red. If nothing ran, write `TODO: add the commands that you ran.` Never write
  a result you did not see.
  - Repo-level PR: the first line is "No plugin folder changed." Then the root checkers and anything else that ran.

## 4. Ask for approval

Open the draft in the developer's editor (`cursor tmp-PR-content.md` or `code tmp-PR-content.md`), or give them its
path. They can edit the file or ask for changes in the chat. Tell them to write `approve` when it is ready, and wait.
`approve` approves this text only: the push and the PR still need their own word (see the skill).

## 5. Read it back

After `approve`, read `tmp-PR-content.md` again from disk. Never send your earlier draft: the developer may have
edited it. Check that line 1 is `# Draft PR Content`, line 2 starts with `Title: ` and line 3 is empty, and put
them back if they moved. If the title is empty, ask for one. If the edit added a claim you can check (a result, a
sha, an issue number), check it and tell the developer about any mismatch before you go on.

## 6. Send it

Only after the developer's word for this step: a word for the push alone does not cover it. Run the block as one
command from the repository root; for an open PR, use `gh pr edit <n>` in place of `gh pr create --base main`:

```sh
title="$(awk 'NR == 2 { sub(/^Title: */, ""); print; exit }' tmp-PR-content.md)"
awk 'NR > 3' tmp-PR-content.md > tmp-PR-body.md
gh pr create --base main --title "$title" --body-file tmp-PR-body.md
```

The shell takes the title from the file, so backticks and `$(...)` in it stay text. A title typed into the command
runs them as shell code inside double quotes and breaks on its own quote marks inside single quotes. GitHub gets
neither draft line.

## 7. Clean up

Remove `tmp-PR-content.md` and `tmp-PR-body.md`, and give the developer the PR's URL.
