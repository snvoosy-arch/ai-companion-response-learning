"""Local versioned Markdown artifacts. No Actor-supplied filesystem paths."""
import json
import re
import stat
from pathlib import Path


class DraftStore:
    def __init__(self, root):
        self.root = Path(root).absolute()
        self._check()
        self.root.mkdir(parents=True, exist_ok=True)

    def _check(self):
        for path in (self.root, *self.root.parents):
            if path.exists() or path.is_symlink():
                info = path.lstat()
                if path.is_symlink() or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                    raise ValueError("redirected draft directory")

    def _valid(self, draft_id):
        return (isinstance(draft_id, str) and re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", draft_id)
                and draft_id.upper() not in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))})

    def _result(self, tool, **values):
        return {"tool": tool, "ok": "error" not in values, "untrusted": True, **values}

    def _save(self, tool, draft_id, content, sources, create):
        self._check()
        if not self._valid(draft_id) or not isinstance(content, str) or not content.strip() or len(content) > 12000:
            return self._result(tool, error="invalid_draft")
        versions = [int(p.name.split(".v")[1].split(".")[0]) for p in self.root.glob(f"{draft_id}.v*.md")
                    if re.fullmatch(re.escape(draft_id) + r"\.v[0-9]+\.md", p.name)]
        if create and versions:
            return self._result(tool, error="draft_exists")
        if not create and not versions:
            return self._result(tool, error="draft_not_found")
        version = max(versions, default=0) + 1
        metadata = {"draft_id": draft_id, "version": version, "sources": sources, "answer_status": "unreviewed"}
        path = self.root / f"{draft_id}.v{version}.md"
        # Exclusive creation prevents overwriting a version or following a file link.
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("<!-- " + json.dumps(metadata, ensure_ascii=True) + " -->\n" + content)
        return self._result(tool, **metadata, filename=path.name)

    def write_draft(self, draft_id, content, sources):
        return self._save("write_draft", draft_id, content, sources, True)

    def revise_draft(self, draft_id, content, sources):
        return self._save("revise_draft", draft_id, content, sources, False)

    def read_draft(self, draft_id, version):
        self._check()
        if not self._valid(draft_id) or type(version) is not int or version < 1:
            return self._result("read_draft", error="invalid_draft")
        path = self.root / f"{draft_id}.v{version}.md"
        if not path.exists():
            return self._result("read_draft", error="draft_not_found")
        info = path.lstat()
        if path.is_symlink() or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            return self._result("read_draft", error="redirected_draft")
        with path.open(encoding="utf-8") as stream:
            text = stream.read(200000)
        header, content = text.split("\n", 1)
        metadata = json.loads(header[5:-4])
        return self._result("read_draft", **metadata, content=content[:12000])
