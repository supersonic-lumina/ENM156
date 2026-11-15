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

* **feat** New functionality
* **fix** Bug fix
* **refactor** Code restructure without behavioural changes
* **test** Adding or improving tests
* **docs** Adding or improving things that are not active code
* **chore** Maintenance, fixing dependencies/tools
* **perf** Optimization, performance improvements
* **ci** CI/CD changes


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

TODO

## Code review

TODO

## Coding standards

TODO

## Testing

TODO

## License

Not merged.

TODO