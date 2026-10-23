# Releasing a plugin

A release is one pull request that bumps the version, then a tag on its merge commit. The steps are in
`docs/writing-a-plugin.md` "Releases"; this file adds what they leave implicit and where you stop. `<folder>` is the
plugin's folder, `<version>` the new version.

## The release PR

1. Check where you are: `git status`, `git branch --show-current`, `git fetch origin`. If you are not on `main`, or
   the tree has uncommitted changes, or `main` has commits that `origin/main` lacks, stop and ask the developer.
   Never switch away from their branch or work on your own. On a clean `main`: `git pull --ff-only`, then
   `git switch -c chore/<folder>-<version>`.
2. In `<folder>/pyproject.toml`, set `version = "<version>"` under `[project]`.
3. From the plugin's folder, run `uv lock`, always, then `uv sync --locked`. The lock records the plugin's own
   version, so a bump without `uv lock` leaves it stale and `uv sync --locked` and the `test` job fail
   (`docs/writing-a-plugin.md` "Versions and dependencies"). Commit `<folder>/uv.lock` with the bump.
4. In `<folder>/CHANGELOG.md`, rename the heading `## [Unreleased]` to `## [<version>]` so its entries sit under the
   new version. No entries under `## [Unreleased]` means there is nothing to release: ask the developer.
5. In the root `README.md`, update the plugin's Status cell in the Plugins table to `<version> on PyPI`. The checker
   does not check this cell, so nothing will remind you.
6. Run the checks, commit, and write the body as for any plugin PR (the skill's steps 3 to 8). `What` names the
   folder and the version and points readers at the version's CHANGELOG section. Push and open only on the
   developer's word.

For a plugin's first release, an org admin must first finish `CONTRIBUTING.md` "For maintainers: a new plugin
folder". Those are the admin's steps: list what is missing and stop.

## Merge

Never merge the release PR, as with any PR: not on the developer's word, not when it is approved and green, not
with `gh pr merge --admin` or any other bypass. The PR merges when a code owner other than its author approves it,
and a maintainer merges it on GitHub. Tell the developer it waits for that review.

## Tag

Only after a maintainer has merged the PR, and only on the developer's word given after the merge. An earlier
"then tag it" is not that word: ask again now.

1. Confirm the merge: `gh pr view <n> --json state,mergeCommit` shows `MERGED` and the merge commit's sha.
2. `git fetch origin`, then check that the merge commit has the new version:
   `git show <sha>:<folder>/pyproject.toml | grep '^version'`.
3. Show the developer both commands and wait for the word:

   ```sh
   git tag <folder>-v<version> <sha>
   git push origin <folder>-v<version>
   ```

4. On the word, run them. The tag starts `.github/workflows/<folder>-release.yml`, which tests, builds and then waits
   for the `pypi-<folder>` environment's reviewer.

## Publish approval

Approving the `pypi-<folder>` deployment is the developer's own action, never yours, even when they ask you to do it
from a phone. That approval is the human decision to put this version on PyPI, recorded under the reviewer's name,
and a published version cannot be replaced. Never call the deployment approval API (`pending_deployments`) and
never re-run the release workflow.

Tell them where to click: on GitHub, the repository's Actions tab, the waiting `<folder>-release` run, then
"Review deployments" (the GitHub mobile app has the same button on the run). Offer a read-only status, for example
`gh run view <run-id>`, and stop.
