"""Tests for the git-sync path (`api.config.git`).

These functions were bug-fixed by inspection and had no tests, which is the state that lets a
fix be wrong in the same direction as the bug. Nothing here touches GitHub: the PyGithub
`Repository` is a hand-written fake whose calls are recorded, so the split-and-retry logic can be
driven at its boundaries -- which is where both defects lived.
"""

import asyncio

import pytest
from github import GithubException

from api.config import git as git_module


class FakeRef:
    def __init__(self):
        self.edited_to = []

    def edit(self, sha):
        self.edited_to.append(sha)


class FakeCommit:
    def __init__(self, sha):
        self.sha = sha


class FakeRepo:
    """Records tree/commit calls and fails the first `fail_times` create_git_tree calls.

    `fail_times` models "the tree is too large for one request", the condition the retry exists
    for -- the only way to reach the split branch without a network.
    """

    def __init__(self, fail_times=0):
        self.fail_times = fail_times
        self.tree_calls = []
        self.commits = []

    def create_git_tree(self, element_list, base_tree):
        self.tree_calls.append(list(element_list))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise GithubException(422, {"message": "tree too large"}, None)
        return f"tree-of-{len(element_list)}"

    def get_git_commit(self, sha):
        return FakeCommit(sha)

    def create_git_commit(self, message, tree, parents):
        self.commits.append((message, tree))
        return FakeCommit(f"commit-{len(self.commits)}")


def _commit(repo, ref, elements):
    return asyncio.run(
        git_module.commit_git_files(repo, ref, "base-sha", "base-tree", "msg", elements)
    )


def test_a_commit_that_succeeds_moves_the_ref_once():
    repo, ref = FakeRepo(), FakeRef()

    _commit(repo, ref, ["a", "b", "c"])

    assert len(repo.commits) == 1
    assert ref.edited_to == ["commit-1"]


def test_an_oversized_tree_is_split_in_half_and_both_halves_commit():
    """The retry's whole purpose: a tree too large for one request succeeds as two."""
    repo, ref = FakeRepo(fail_times=1), FakeRef()

    _commit(repo, ref, ["a", "b", "c", "d"])

    # First call carried all four and failed; the halves then went as two separate trees.
    assert [len(c) for c in repo.tree_calls] == [4, 2, 2]
    assert len(repo.commits) == 2


def test_a_single_unsplittable_element_raises_instead_of_reporting_success():
    """The bug this file exists for.

    A list of one cannot be split -- `1 // 2` is 0 -- so the old code fell through to
    `print_exc()` and returned as though it had committed. A scheduled data dump lost that file
    silently, and the caller could not tell a completed dump from a partial one.
    """
    repo, ref = FakeRepo(fail_times=99), FakeRef()

    with pytest.raises(GithubException):
        _commit(repo, ref, ["only-one"])

    assert repo.commits == []
    assert ref.edited_to == [], "the ref must not move when the commit failed"


def test_an_empty_element_list_does_not_raise():
    """`len([]) // 2` is also 0, so the unsplittable branch must not fire on nothing to do."""
    repo, ref = FakeRepo(), FakeRef()

    _commit(repo, ref, [])

    assert len(repo.commits) == 1


def test_the_default_commit_message_is_built_per_call_not_at_import(monkeypatch):
    """A default argument is evaluated once, at import, so the old default froze the timestamp.

    Every commit from a long-running process would have claimed the same time. Asserted by
    varying the clock between two calls and reading the message the repo actually received.
    """
    stamps = iter(["11:00:00 01-01-2026", "12:00:00 01-01-2026"])

    class FrozenDatetime:
        @staticmethod
        def now():
            class _N:
                @staticmethod
                def strftime(_fmt):
                    return next(stamps)

            return _N()

    monkeypatch.setattr(git_module, "datetime", FrozenDatetime)

    seen = []

    async def record(repo, ref, sha, tree, message, elements):
        seen.append(message)

    monkeypatch.setattr(git_module, "commit_git_files", record)
    monkeypatch.setattr(git_module.g, "get_repository", lambda name: FakeRepoWithRefs())

    asyncio.run(git_module.update_git_files([], [], "repo", "master"))
    asyncio.run(git_module.update_git_files([], [], "repo", "master"))

    assert (
        seen[0] != seen[1]
    ), "two calls produced the same timestamp - the default froze"


class FakeRefObject:
    sha = "base-sha"


class FakeRepoWithRefs(FakeRepo):
    def get_git_ref(self, _name):
        ref = FakeRef()
        ref.object = FakeRefObject()
        return ref

    def get_git_tree(self, _sha):
        return "base-tree"
