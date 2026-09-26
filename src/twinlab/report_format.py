"""Number report sections and validate the local links shipped in the archive."""
from pathlib import Path
import re
from urllib.parse import unquote


def slug(title):
    return re.sub(r"[^\w -]", "", title.lower()).replace(" ", "-")


def headings(text):
    fenced = False
    for index, line in enumerate(text.splitlines()):
        if line.startswith("```"):
            fenced = not fenced
        if not fenced and (match := re.match(r"^(#{1,4}) (.+)$", line)):
            yield index, len(match[1]), match[2]


def number_sections(root):
    """Idempotent numbering with matching updates to cross-chapter anchors."""
    root = Path(root)
    documents = [root / "README.md", *sorted((root / "docs").glob("0[1-6]_*.md"))]
    texts, anchors = {}, {}
    for path in documents:
        text = path.read_text()
        lines = text.splitlines()
        if path.name != "README.md":
            chapter = int(path.name[:2])
            section = subsection = 0
            for index, level, title in headings(text):
                if level not in (2, 3):
                    continue
                plain = re.sub(r"^\d+(?:\.\d+)+\.?\s+", "", title)
                if level == 2:
                    section += 1
                    subsection = 0
                    number = f"{chapter}.{section}"
                else:
                    subsection += 1
                    number = f"{chapter}.{section}.{subsection}"
                new_title = f"{number} {plain}"
                for old in (title, plain):
                    anchors[path.resolve(), slug(old)] = slug(new_title)
                lines[index] = "#"*level+" "+new_title
        texts[path] = "\n".join(lines)+"\n"
    for path, text in texts.items():
        def rewrite(match):
            target, separator, anchor = match[1].partition("#")
            if not separator or target.startswith(("https:", "http:", "mailto:")):
                return match[0]
            destination = (path.parent/unquote(target)).resolve() if target else path.resolve()
            new = anchors.get((destination, unquote(anchor)), anchor)
            return f"]({target}#{new})"
        path.write_text(re.sub(r"\]\(([^)]+)\)", rewrite, text))


def validate_links(files):
    included = {p.resolve() for p in files}
    for path in files:
        if path.suffix != ".md":
            continue
        text = path.read_text()
        if sum(line.startswith("```") for line in text.splitlines()) % 2:
            raise ValueError(f"Unclosed Markdown fence: {path}")
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            if target.startswith(("http:", "https:", "mailto:")):
                continue
            filename, separator, fragment = unquote(target).partition("#")
            destination = (path.parent/filename).resolve() if filename else path.resolve()
            if destination not in included:
                raise ValueError(f"Submission link is missing: {path.name}: {target}")
            if separator and fragment and destination.suffix == ".md":
                targets = {slug(title) for _, _, title in headings(destination.read_text())}
                if fragment not in targets:
                    raise ValueError(f"Submission anchor is missing: {path.name}: {target}")


def replace_section(text, start_title, end_title, replacement):
    """Replace generated content without depending on its section number."""
    pattern = lambda title: rf"(?m)^## (?:\d+(?:\.\d+)+ )?{re.escape(title)}[^\n]*\n"
    start = re.search(pattern(start_title), text)
    if not start:
        raise ValueError(f"Missing generated section: {start_title}")
    end = re.search(pattern(end_title), text[start.end():])
    if not end:
        raise ValueError(f"Missing following section: {end_title}")
    stop = start.end()+end.start()
    return text[:start.start()]+replacement.rstrip()+"\n\n"+text[stop:]
