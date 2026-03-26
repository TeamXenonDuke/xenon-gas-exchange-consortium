from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple


# ----------------- ANSI color helpers -----------------
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _supports_color() -> bool:
    """
    Return True if ANSI colors likely work for this process output.
    (Logging usually goes to stderr, but we check both stderr/stdout.)
    Respects NO_COLOR if set.
    """
    try:
        is_tty = sys.stderr.isatty() or sys.stdout.isatty()
        term_ok = os.environ.get("TERM", "") not in ("", "dumb")
        no_color = os.environ.get("NO_COLOR") is not None
        return is_tty and term_ok and (not no_color)
    except Exception:
        return False


def _red(s: str) -> str:
    return f"{RED}{s}{RESET}" if _supports_color() else s


def _yellow(s: str) -> str:
    return f"{YELLOW}{s}{RESET}" if _supports_color() else s


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _max_line_len(messages: List[str]) -> int:
    """Longest visible line length across all messages (ANSI stripped)."""
    m = 0
    for msg in messages:
        for line in _strip_ansi(msg).splitlines() or [""]:
            m = max(m, len(line))
    return m


# ----------------- git runner -----------------
def _run_git(repo_dir: str, *args: str, check: bool = True) -> str:
    """Run a git command inside repo_dir and return stdout (stripped)."""
    res = subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{res.stderr.strip()}")
    return res.stdout.strip()


def _fetch(repo_dir: str) -> None:
    """Refresh remote refs; ignore failures (offline, auth, etc.)."""
    try:
        _run_git(repo_dir, "fetch", "--quiet", "--all", "--prune", check=True)
    except Exception:
        pass


def _remote_default_branch(repo_dir: str, remote: str = "origin") -> Optional[str]:
    """
    Return remote default branch ref (e.g., "origin/main") using refs/remotes/origin/HEAD.
    Fallback to origin/main then origin/master if present.
    """
    try:
        ref = _run_git(
            repo_dir,
            "symbolic-ref",
            "--quiet",
            "--short",
            f"refs/remotes/{remote}/HEAD",
            check=False,
        )
        if ref:
            return ref  # e.g. "origin/main"
    except Exception:
        pass

    for cand in (f"{remote}/main", f"{remote}/master"):
        try:
            _run_git(repo_dir, "show-ref", "--verify", "--quiet", f"refs/remotes/{cand}", check=True)
            return cand
        except Exception:
            continue

    return None


def _ref_exists(repo_dir: str, ref: str) -> bool:
    """True if ref can be resolved (branch/tag/commit)."""
    try:
        _run_git(repo_dir, "rev-parse", "--verify", "--quiet", ref, check=True)
        return True
    except Exception:
        return False


def _ahead_behind(repo_dir: str, left_ref: str, right_ref: str) -> Tuple[int, int]:
    """
    Return (ahead, behind) of left_ref relative to right_ref.
      - ahead  = commits only in left_ref
      - behind = commits only in right_ref
    """
    counts = _run_git(repo_dir, "rev-list", "--left-right", "--count", f"{left_ref}...{right_ref}")
    left, right = counts.split()
    return int(left), int(right)


def _log_oneline(repo_dir: str, rev_range: str, n: int) -> str:
    """Return `git log --oneline <rev_range> -n<n>` output."""
    return _run_git(repo_dir, "log", "--oneline", rev_range, f"-n{n}", check=False).strip()


def _extract_merge_lines(oneline_log: str) -> List[str]:
    """Extract lines that look like merge commits / PR merges."""
    out: List[str] = []
    for line in oneline_log.splitlines():
        low = line.lower()
        if "merge pull request" in low or low.startswith("merge ") or "merge branch" in low:
            out.append(line)
    return out


# ----------------- repo state -----------------
@dataclass
class RepoState:
    """Snapshot of local repo status + HEAD info."""
    repo_dir: str
    branch: str
    head_sha: str
    head_subject: str
    dirty: bool
    untracked: bool
    unmerged_files: List[str]
    in_merge: bool
    in_rebase: bool
    in_cherry_pick: bool


def get_repo_state(repo_dir: str = ".") -> RepoState:
    """Collect local repo status (no network)."""
    repo_dir = os.path.abspath(repo_dir)

    _run_git(repo_dir, "rev-parse", "--is-inside-work-tree")

    branch = _run_git(repo_dir, "rev-parse", "--abbrev-ref", "HEAD")
    head_sha = _run_git(repo_dir, "rev-parse", "HEAD")
    head_subject = _run_git(repo_dir, "log", "-1", "--pretty=%s")

    porcelain = _run_git(repo_dir, "status", "--porcelain")
    dirty = len(porcelain) > 0
    untracked = any(line.startswith("??") for line in porcelain.splitlines())

    unmerged = _run_git(repo_dir, "diff", "--name-only", "--diff-filter=U")
    unmerged_files = [x for x in unmerged.splitlines() if x.strip()]

    git_dir = _run_git(repo_dir, "rev-parse", "--git-dir")
    git_dir = os.path.join(repo_dir, git_dir) if not os.path.isabs(git_dir) else git_dir

    in_merge = os.path.exists(os.path.join(git_dir, "MERGE_HEAD"))
    in_rebase = (
        os.path.exists(os.path.join(git_dir, "rebase-apply"))
        or os.path.exists(os.path.join(git_dir, "rebase-merge"))
    )
    in_cherry_pick = os.path.exists(os.path.join(git_dir, "CHERRY_PICK_HEAD"))

    return RepoState(
        repo_dir=repo_dir,
        branch=branch,
        head_sha=head_sha,
        head_subject=head_subject,
        dirty=dirty,
        untracked=untracked,
        unmerged_files=unmerged_files,
        in_merge=in_merge,
        in_rebase=in_rebase,
        in_cherry_pick=in_cherry_pick,
    )


# ----------------- main checker -----------------
def warn_git_status(
    repo_dir: str = ".",
    do_fetch: bool = True,
    show_n: int = 5,
    compare_branch: Optional[str] = None,
    git_always_show: bool = True,
) -> None:
    """
    Log a “git health check” for the repository.

    Behavior
    --------
    - Compares HEAD against `compare_branch` (default: remote default branch like origin/main).
    - Emits WARNING (red) only when:
        * you are BEHIND `compare_branch`, or
        * `compare_branch` cannot be resolved.
    - Being AHEAD of `compare_branch` is INFO-only (no warning).
    - Shows incoming/outgoing commits in YELLOW when displayed.

    Parameters
    ----------
    repo_dir:
        Path to the repo. "." = current directory.
    do_fetch:
        If True, runs `git fetch --all --prune` (best effort) first.
    show_n:
        Max commit lines to show for incoming/outgoing.
    compare_branch:
        Ref to compare against (e.g., "origin/main").
        If None, auto-detects origin/HEAD (fallback origin/main/origin/master).
        If you pass "main" or "some-branch" (no "/"), it auto-prefixes "origin/".
    git_always_show:
        - True  -> always log header/status.
        - False -> only log when there is a compare-branch WARNING (behind/missing ref).
    """
    log = logging.getLogger("git-check")
    repo_dir = os.path.abspath(repo_dir)

    if do_fetch:
        _fetch(repo_dir)

    st = get_repo_state(repo_dir)

    issues: List[str] = []         # all issues we will print if output is enabled
    compare_issues: List[str] = [] # only issues that should trigger output when git_always_show=False

    # Decide what to compare against
    if compare_branch is None:
        compare_branch = _remote_default_branch(repo_dir, remote="origin")
    else:
        compare_branch = compare_branch.strip()
        if compare_branch and ("/" not in compare_branch) and (compare_branch not in ("HEAD",)):
            compare_branch = f"origin/{compare_branch}"

    if not compare_branch or not _ref_exists(repo_dir, compare_branch):
        msg = f"Cannot find compare_branch: {compare_branch!r}"
        issues.append(msg)
        compare_issues.append(msg)
        compare_branch = None

    # Prepare log lines first (so separator matches the longest line)
    info_lines: List[str] = [
        f"[git-check] Current branch: {st.branch}",
        f"[git-check] HEAD: {st.head_sha[:8]} — {st.head_subject}",
    ]
    detail_blocks: List[str] = []  # incoming/outgoing details (YELLOW)

    behind: Optional[int] = None
    ahead: Optional[int] = None

    if compare_branch:
        ahead, behind = _ahead_behind(repo_dir, "HEAD", compare_branch)

        if behind == 0 and ahead == 0:
            info_lines.append(f"[git-check] Up to date with {compare_branch}")
        else:
            # BEHIND => WARNING (triggers output when git_always_show=False)
            if behind > 0:
                msg = f"Behind {compare_branch} by {behind} commit(s). (You need to pull/rebase)"
                issues.append(msg)
                compare_issues.append(msg)

                incoming = _log_oneline(repo_dir, f"HEAD..{compare_branch}", show_n)
                if incoming:
                    detail_blocks.append(
                        _yellow(f"[git-check] Incoming from {compare_branch} (would be pulled):\n{incoming}")
                    )
                    merges = _extract_merge_lines(incoming)
                    if merges:
                        detail_blocks.append(
                            _yellow("[git-check] Merge/PR commits among incoming:\n" + "\n".join(merges))
                        )

            # AHEAD => INFO only (does NOT trigger output when git_always_show=False)
            if ahead > 0:
                info_lines.append(
                    f"[git-check] Ahead of {compare_branch} by {ahead} commit(s). (Local commits not in {compare_branch})"
                )
                outgoing = _log_oneline(repo_dir, f"{compare_branch}..HEAD", show_n)
                if outgoing:
                    detail_blocks.append(
                        _yellow(f"[git-check] Outgoing vs {compare_branch} (would be pushed):\n{outgoing}")
                    )

    # Only show output if requested and compare-branch issues exist
    if (not git_always_show) and (len(compare_issues) == 0):
        return

    # Separator length = longest visible line in anything we will log
    all_for_len: List[str] = []
    all_for_len.extend(info_lines)
    all_for_len.extend(detail_blocks)
    if issues:
        all_for_len.append("[git-check][WARNING]")
        all_for_len.extend([f"  - {m}" for m in issues])
        if compare_branch and behind and behind > 0:
            all_for_len.append("  Hint: update with: git pull --rebase  (or git pull)")

    sep = "#" * max(1, _max_line_len(all_for_len))

    # Beginning separator
    log.info(sep)

    # Header
    for line in info_lines:
        log.info(line)

    # WARNING summary first (RED)
    if issues:
        log.warning(_red("[git-check][WARNING]"))
        for m in issues:
            log.warning(_red(f"  - {m}"))

    # Details after (YELLOW)
    for blk in detail_blocks:
        log.warning(blk)

    # Hints (YELLOW) — only meaningful for BEHIND
    if compare_branch and behind and behind > 0:
        log.warning(_yellow("  Hint: update with: git pull --rebase  (or git pull)"))
    elif not issues:
        log.info("[git-check] OK (no warnings).")

    # Ending separator
    log.info(sep)

