# Contributing to ENM156

This guide explains the git workflow of this repo.

## Table of contents
- [Getting started](#getting-started)
- [Workflow overview](#workflow-overview)
- [Branches](#branches)
- [Commits](#commits)
- [Pull requests](#pull-requests)
- [Code review](#code-review)
- [Coding standards](#coding-standards)
- [Testing](#testing)
- [Reporting bugs and security issues](#reporting-bugs-and-security-issues)
- [License](#license)

## Getting started

Clone the repo and follow the setup instructions in the [README](README.md).

## Workflow overview

1. **Find or create an issue.** Use the [Planning Board](https://github.com/orgs/supersonic-lumina/projects/1) to grab one or create one using `.github/ISSUE_TEMPLATE` (TODO: does not exist yet).
2. **Assign it to yourself** and move the ticket to **In progress** on the [Planning Board](https://github.com/orgs/supersonic-lumina/projects/1).
3. **Create a branch from the issue** `gh issue develop [#] --name [name]` or `git checkout -b [name]` and manually link it. Check out [branch naming conventions](#branch-format).
4. **Open a draft PR early** if you want feedback, and mark it **Ready for review** when it's done.
5. **Move the ticket to In review** once the PR is ready.
6. **Address review feedback.** The ticket goes back to **In progress** if changes are requested.
7. **Merge** after approval. The issue closes automatically and moves to **Done**.

You are not required to assign an issue to yourself, finding and creating one is good work too.

Not sure where to start? Look for issues labeled `good first issue`.

Which parts of workflow do what:
* **Branch name** determines the type of work (feat, fix, chore, etc.). [Read more](#branch-format)
* **Issue** describes what needs to be done and potential proposals for doing that
* **Commit messages** describe individual changes made in that commit, freeform
* **PR title** follows [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) because it becomes the final commit on main through [squash merging](#pull-requests)

## Branches

### Branch Types

| Type | Description |
|-:|:-|
| **feat** | New feature |
| **fix** | Bug fix |
| **refactor** | Code restructure without behavioural changes |
| **test** | Adding or improving tests |
| **docs** | Adding or improving things that are not active code |
| **chore** | Maintenance, fixing dependencies/tools |
| **perf** | Optimization, performance improvements |
| **ci** | CI/CD changes |


### Branch Format

```
[type]/[issue#]-[description]
```
where "-" replaces any spacebar.


Example:
```
feat/45-membrane-potential-visualization
```

### Other Branch Conventions

- Branch from `main` and keep branches short-lived. Merge within a few days where possible.
- Never push directly to `main`, all changes go through a PR.
- One issue per branch and one branch per PR. If you discover unrelated work, open a new issue.

## Commits

We use squash-merge so the **PR title** becomes the final commit message on `main`, make it follow [Conventional Commits](https://www.conventionalcommits.org/). Individual commits inside your branch can be messier, but keep them meaningful and to the point.

## Pull requests

PR titles use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(optional scope): <short summary in imperative mood>
```

- Fill in the PR template completely, especially `Closes #<issue>` (TODO: does not exist yet).
- Keep PRs small and focused for a meaningful review process. Aim for under about 400 changed lines. Split large work into smaller PRs.
- Open as **Draft** while work is in progress.
- Preferrably update tests and documentation with your code, not afterwards.
- Don't mix refactoring or formatting changes with functional changes.

### Definition of done

Checklist for when a PR can be merged:
- [ ] At least 1 approving review
- [ ] All review feedback implemented
- [ ] All comments are reasonably responded to and/or resolved
- [ ] The PR is linked to an issue which it resolves

Only **Squash and merge** is allowed to keep commit history clean. The branch is deleted automatically.

## Code review

**Writing code**
- Review your own diff first
- Respond to every comment. Resolve and leave a reaction when addressed, or explain why you disagree.
- Don't take feedback and rejections personally. Reviews are about code only.

**Reviewing code**
- Aim to respond whithin one working day. (excluding weekends I guess)
- Label optional suggestions (nitpicks) with `nit:` so authors know what is necessary and what is a recommendation
- Explain *why* behing each request. Suggest alternatives when able.
- Approve when it works as it should and not when its done exactly like you would've.
- Check not only for correctness, but also for readability, security and whether the work done is relevant to the issue at hand.
- If work, unrelated to the issue was done in a PR, reject and request for it to be split into a separate issue.

## Coding standards

General conventions:
- **Naming:** Use `snake_case` for variables, functions, and modules; `PascalCase` for classes and types; and `UPPER_SNAKE_CASE` for constants. Prefer descriptive but longer names over abbreviations.
- **Type hints:** Add type annotations to function parameters and return values. Only use `Any` if absolutely necessary.
- **Error handling:** Never silently swallow exceptions, always specify the excpetion kind explicitly.
- **Dependencies:** Always notify the team before introducing new dependencies. Make sure that their license is compatible with `AGPL-3.0`
- **Comments and documentation:** Prefer self-documenting code and avoid unnecessary comments that restate the obvious.
- **Functions and classes:** Keep them small, focused on a single responsibility
- **Mutability:** Avoid mutable state when possible. Always prefer immutable data structures.
- **Imports:** Avoid wildcard imports when practical.
- **Configuration:** Keep your local configuration outside of the source code. If you add a local config, make sure its gitignored

## Testing

TODO

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE). By submitting a contribution, you agree that it is licensed under the same terms. Only submit code you wrote or have the right to submit under these terms.