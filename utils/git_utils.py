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

    Parameters
    ----------
    repo_dir:
        Path to the repo. "." = current directory.
    do_fetch:
        If True, runs `git fetch --all --prune` (best effort) first.
    show_n:
        Max commit lines to show for incoming/outgoing.
    compare_branch:
        The branch/ref to compare against (default: remote default branch like origin/main).
        Examples:
            - "origin/main"
            - "origin/redo-fv-and-99-separate"
            - "redo-fv-and-99-separate"  (auto-prefixes "origin/")
        If None, auto-detects origin/HEAD (fallback origin/main or origin/master).
    git_always_show:
        - True  -> always log header/status
        - False -> only log if there is a compare-branch related warning
                  (behind/ahead compare_branch, or cannot find compare_branch).
                  Local-only dirtiness won't trigger output by itself.
    """
    log = logging.getLogger("git-check")
    repo_dir = os.path.abspath(repo_dir)

    if do_fetch:
        _fetch(repo_dir)

    st = get_repo_state(repo_dir)

    issues: List[str] = []
    compare_issues: List[str] = []

    # Local state warnings
    if st.in_merge:
        issues.append("Merge in progress (MERGE_HEAD exists).")
    if st.in_rebase:
        issues.append("Rebase in progress.")
    if st.in_cherry_pick:
        issues.append("Cherry-pick in progress.")
    if st.unmerged_files:
        issues.append(f"Unmerged conflict files: {', '.join(st.unmerged_files)}")
    if st.dirty:
        issues.append("Working tree has local changes (uncommitted and/or untracked).")

    # Decide what to compare against
    if compare_branch is None:
        compare_branch = _remote_default_branch(repo_dir, remote="origin")  # e.g., origin/main
    else:
        compare_branch = compare_branch.strip()
        if compare_branch and ("/" not in compare_branch) and (compare_branch not in ("HEAD",)):
            compare_branch = f"origin/{compare_branch}"

    if not compare_branch or not _ref_exists(repo_dir, compare_branch):
        msg = f"Cannot find compare_branch: {compare_branch!r}"
        issues.append(msg)
        compare_issues.append(msg)
        compare_branch = None

    # Collect log lines first (so separator matches the longest line)
    info_lines: List[str] = []
    detail_blocks: List[str] = []  # incoming/outgoing details (YELLOW)

    info_lines.append(f"[git-check] Current branch: {st.branch}")
    info_lines.append(f"[git-check] HEAD: {st.head_sha[:8]} — {st.head_subject}")

    # Compare HEAD vs compare_branch
    if compare_branch:
        ahead, behind = _ahead_behind(repo_dir, "HEAD", compare_branch)

        if behind == 0 and ahead == 0:
            info_lines.append(f"[git-check] Up to date with {compare_branch}")
        else:
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

            if ahead > 0:
                msg = f"Ahead of {compare_branch} by {ahead} commit(s). (You have local commits not in {compare_branch})"
                issues.append(msg)
                compare_issues.append(msg)

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
        if compare_branch:
            all_for_len.append("  Hint: update with: git pull --rebase  (or git pull)")
        if st.dirty:
            all_for_len.append("  Hint: see local changes: git status")
        if st.unmerged_files:
            all_for_len.append("  Hint: resolve conflicts then continue merge/rebase")

    sep = "#" * max(1, _max_line_len(all_for_len))

    # Beginning separator
    log.info(sep)

    # Header
    for line in info_lines:
        log.info(line)

    # ✅ Put RED warning first (before Incoming/Outgoing)
    if issues:
        log.warning(_red("[git-check][WARNING]"))
        for m in issues:
            log.warning(_red(f"  - {m}"))

    # ✅ Then show Incoming/Outgoing in YELLOW
    for blk in detail_blocks:
        log.warning(blk)

    # Hints (YELLOW)
    if issues:
        if compare_branch:
            log.warning(_yellow("  Hint: update with: git pull --rebase  (or git pull)"))
        if st.dirty:
            log.warning(_yellow("  Hint: see local changes: git status"))
        if st.unmerged_files:
            log.warning(_yellow("  Hint: resolve conflicts then continue merge/rebase"))
    else:
        log.info("[git-check] OK (no warnings).")

    # Ending separator
    log.info(sep)
