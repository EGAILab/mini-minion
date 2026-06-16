"""ApplyPatchTool — apply a unified diff patch to one or more files.

Why this tool?
--------------
:class:`PatchPreviewTool` exists to *show* what a diff would do without
touching the filesystem.  :class:`ApplyPatchTool` is the companion that
actually *applies* the diff.

The tool shells out to ``git apply`` (via the same ``_git`` helper from
``git.py``) because:

- ``git apply`` understands unified diff format with full file headers.
- It handles multi-file patches atomically — it either applies cleanly or
  reports every rejected hunk rather than silently applying partial changes.
- It works on any directory, not just in the repo root.
- The ``--check`` flag lets us do a dry run to catch failures before touching
  files on disk.

Talks to
--------
- ``base.py`` — extends :class:`Tool`, returns :class:`ToolSchema`.
- ``policy.py`` — calls ``check_write()`` before writing to disk.
- ``registry.py`` / ``__init__.py`` — registered via ``default_registry()``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .base import Tool, ToolSchema
from .policy import PermissionPolicy

# Cached result of shutil.which("git") so we don't re-run it on every call.
# None means git is not on PATH; a non-None string is the full path to the binary.
_GIT_PATH: str | None = shutil.which("git")

_DEFAULT_TIMEOUT = 30  # seconds


class ApplyPatchTool(Tool):
    """Apply a unified diff patch string to one or more files.

    Uses ``git apply`` under the hood so it handles multi-file patches,
    context lines, and renamed files correctly.
    """

    def __init__(
        self,
        cwd: Path | None = None,
        policy: PermissionPolicy | None = None,
    ) -> None:
        # cwd: directory where git apply runs; typically the workspace root.
        self._cwd = cwd
        # policy: guards against read_only_mode and workspace escapes.
        self._policy = policy

    @property
    def schema(self) -> ToolSchema:
        """Describe this tool to the LLM."""
        return ToolSchema(
            name="apply_patch",
            description=(
                "Apply a unified diff patch to one or more files using git apply. "
                "The patch must be in standard unified diff format with --- and +++ headers. "
                "Returns the git apply output on success or a detailed error on failure."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": "The unified diff patch string to apply.",
                    },
                    "check_only": {
                        "type": "boolean",
                        "description": (
                            "If true, verify the patch applies cleanly without "
                            "changing any files (dry run). Default false."
                        ),
                    },
                },
                "required": ["patch"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        """Apply a patch string via git apply.

        Args:
            patch (str): Unified diff patch text.
            check_only (bool, optional): Dry run — verify without applying.

        Returns:
            str: Success message, or an error string describing the failure.
        """
        patch_text = str(kwargs["patch"])
        check_only = bool(kwargs.get("check_only", False))

        # Fail fast if git is not available — avoid creating a temp file just to
        # hit a FileNotFoundError inside subprocess.run.  shutil.which() was
        # called at import time and cached; re-check so tests can monkeypatch it.
        if _GIT_PATH is None and shutil.which("git") is None:
            return (
                "Error: ApplyPatchTool requires git to be installed and on PATH. "
                "Install git or use WriteTool + EditTool for individual file edits."
            )

        # Policy check: writing to disk requires a non-read-only policy.
        # check_only is read-only — skip the check in that case.
        if not check_only and self._policy is not None:
            cwd_path = self._cwd or Path.cwd()
            error = self._policy.check_write(cwd_path)
            if error:
                return error

        # Write the patch text to a temp file; git apply reads from a file.
        # NamedTemporaryFile with delete=False because git apply needs to
        # open the file itself by path on Windows.
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".patch", delete=False, encoding="utf-8"
            ) as f:
                f.write(patch_text)
                patch_path = f.name
        except OSError as exc:
            return f"Error: could not write temp patch file: {exc}"

        try:
            args = ["git", "apply"]
            if check_only:
                args.append("--check")
            args.append(patch_path)

            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=_DEFAULT_TIMEOUT,
                cwd=self._cwd,
            )
            output = (result.stdout + result.stderr).strip()
            if result.returncode != 0:
                return f"Error: git apply failed:\n{output}" if output else "Error: git apply failed (no output)."
            if check_only:
                return "Patch applies cleanly (dry run — no files changed)."
            return f"Patch applied successfully.\n{output}" if output else "Patch applied successfully."
        except subprocess.TimeoutExpired:
            return f"Error: git apply timed out after {_DEFAULT_TIMEOUT} seconds."
        except FileNotFoundError:
            return "Error: git is not installed or not found on PATH."
        except Exception as exc:
            return f"Error: {exc}"
        finally:
            # Clean up the temporary patch file.
            try:
                Path(patch_path).unlink(missing_ok=True)
            except Exception:
                pass
