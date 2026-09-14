# List of workflows and actions
This folder contains workflows that are helpful for maintaining a smooth and secure development process. The workflows should be enabled for open-source projects.

Workflows:
1. `lint-format.yml` - This workflow will run the linter and formatter.
2. `protected-manifest.yml` - This workflow will fail PRs which have change to pyproject,toml (unless contains a label: maintainer-approved-pyproject-change).
3. `qcom-preflight-checks.yml` - This workflow runs several preflight checks, including copyight, email, repolinter, and security checks.  See [qualcomm/qcom-actions](https://github.com/qualcomm/qcom-actions)
4. `stale-issues.yaml` - This workflow will run daily and mark inactive Issues and PRs as stale with a comment and a label. It's currently configured to mark PRs and Issues as stale after 30 days. PRs are then auto-closed after an additional 5 days of inactivity. Adjust this workflow as needed.
5. `unit-tests.yml` - This workflow will run unit tests.
