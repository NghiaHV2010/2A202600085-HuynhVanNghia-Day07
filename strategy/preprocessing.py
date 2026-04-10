from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify as md

INPUT_DIR = Path("data/yte")
OUTPUT_DIR = Path("data/yte_md")

TOC_PATTERN = re.compile(r"<h3>\s*Mục lục\s*</h3>.*?(?=<h2[^>]*>|<h1[^>]*>)", re.IGNORECASE | re.DOTALL)

TAIL_HTML_MARKERS = [
    "<h4><strong>hệ thống bệnh viện đa khoa tâm anh",
    "<strong>hệ thống bệnh viện đa khoa tâm anh",
    "để được tư vấn và",
    "để được tư vấn, giải đáp thắc mắc về thai sản",
    "để đặt lịch khám",
    "để đặt lịch thăm khám",
    "khoa nội thần kinh bệnh viện đa khoa tâm anh",
    "trung tâm tiết niệu thận học, khoa nam học",
    "đồng hành cùng mẹ trong suốt giai đoạn thai kỳ",
    "thai sản trọn gói",
    "gọi tổng đài",
]

REMOVE_LINE_PATTERNS = [
    re.compile(r"^.*https?://\S+.*$", re.IGNORECASE),
    re.compile(r"^\s*Trang chủ\s*>.*$", re.IGNORECASE),
    re.compile(r"^\s*CHUYÊN MỤC BỆNH HỌC\s*>.*$", re.IGNORECASE),
    re.compile(r"^\s*[#>*_`\- ]*Fanpage\s*:?.*$", re.IGNORECASE),
    re.compile(r"^\s*[#>*_`\- ]*Website\s*:?.*$", re.IGNORECASE),
    re.compile(r"^\s*[#>*_`\- ]*Hotline\s*:?.*$", re.IGNORECASE),
    re.compile(r"^\s*[#>*_`\- ]*TP\.HCM\s*:?.*$", re.IGNORECASE),
    re.compile(r"^\s*[#>*_`\- ]*Hà Nội\s*:?.*$", re.IGNORECASE),
    re.compile(r"^\s*Để được tư vấn.*$", re.IGNORECASE),
    re.compile(r"^\s*Để đặt lịch khám.*$", re.IGNORECASE),
    re.compile(r"^\s*Để đặt lịch thăm khám.*$", re.IGNORECASE),
    re.compile(r"^\s*Đăng ký hẹn khám.*$", re.IGNORECASE),
    re.compile(r"^\s*Gửi tin nhắn.*$", re.IGNORECASE),
    re.compile(r"^\s*Nhắn tin qua Zalo.*$", re.IGNORECASE),
    re.compile(r"^\s*108\s+Hoàng\s+Như\s+Tiếp.*$", re.IGNORECASE),
    re.compile(r"^\s*2B\s+Phổ\s+Quang.*$", re.IGNORECASE),
]

REMOVE_CONTAINS_KEYWORDS = [
    "đặt lịch khám",
    "danh-cho-khach-hang/dat-lich-kham",
]

PHONE_PATTERN = re.compile(r"\b0\d{2,3}[\s.-]?\d{3,4}[\s.-]?\d{3,4}\b")

INLINE_TRUNCATE_MARKERS = [
    "trung tâm nội soi và phẫu thuật nội soi tiêu hóa",
    "hệ thống bvđk tâm anh quy tụ",
    "bệnh viện đa khoa tâm anh có các dịch vụ",
    "bên cạnh đó, bệnh viện đa khoa tâm anh",
    "**khoa nội tiết – đái tháo đường**",
]


def strip_leading_noise(raw_html: str) -> str:
    first_h1_index = raw_html.find("<h1")
    if first_h1_index >= 0:
        return raw_html[first_h1_index:]
    return raw_html


def drop_toc_block(raw_html: str) -> str:
    return TOC_PATTERN.sub("", raw_html)


def strip_trailing_noise(raw_html: str) -> str:
    lowered = raw_html.lower()
    cut_indices: list[int] = []
    for marker in TAIL_HTML_MARKERS:
        index = lowered.find(marker)
        if index >= 0:
            cut_indices.append(index)

    if not cut_indices:
        return raw_html

    return raw_html[: min(cut_indices)]


def should_remove_line(line: str) -> bool:
    compact = line.strip()
    if not compact:
        return False

    if PHONE_PATTERN.search(compact):
        return True

    for pattern in REMOVE_LINE_PATTERNS:
        if pattern.match(compact):
            return True

    lowered = compact.lower()
    for keyword in REMOVE_CONTAINS_KEYWORDS:
        if keyword.lower() in lowered:
            return True

    return False


def truncate_inline_promotional_fragment(line: str) -> str:
    lowered = line.lower()
    cut_positions: list[int] = []

    for marker in INLINE_TRUNCATE_MARKERS:
        index = lowered.find(marker)
        if index >= 0:
            cut_positions.append(index)

    if not cut_positions:
        return line

    head = line[: min(cut_positions)].rstrip(" -,:;|")
    return head


def normalize_markdown(md_text: str) -> str:
    md_text = md_text.replace("\r\n", "\n")

    cleaned_lines: list[str] = []
    for line in md_text.split("\n"):
        line = re.sub(r"\s+", " ", line).rstrip()
        line = truncate_inline_promotional_fragment(line)
        if not line.strip():
            continue

        if should_remove_line(line):
            continue

        # Remove breadcrumb fragments and decorative separators.
        if line.strip() in {">", "-", "*"}:
            continue

        cleaned_lines.append(line)

    # Drop trailing marketing tail from the first strong marker if still present.
    tail_markers = [
        "Trung tâm Tiết niệu Thận học, Khoa Nam học",
        "HỆ THỐNG BỆNH VIỆN ĐA KHOA TÂM ANH",
    ]
    for marker in tail_markers:
        for index, line in enumerate(cleaned_lines):
            if marker.lower() in line.lower():
                cleaned_lines = cleaned_lines[:index]
                break


    merged: list[str] = []
    for line in cleaned_lines:
        stripped = line.strip()
        is_heading = stripped.startswith("#")

        if is_heading:
            # Headings always start a new block
            merged.append("")       # blank line before heading
            merged.append(stripped)
            merged.append("")       # blank line after heading
        elif merged and merged[-1] != "" and not merged[-1].startswith("#"):
            # Continue the current paragraph — join with a space
            merged[-1] = merged[-1] + " " + stripped
        else:
            merged.append(stripped)

        text = "\n".join(merged)


    # Keep heading spacing and collapse excess blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)

    # Normalize common punctuation spacing.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([\[(])\s+", r"\1", text)
    text = re.sub(r"\s+([\])])", r"\1", text)

    return text.strip() + "\n"


def convert_one_file(source_path: Path, target_path: Path) -> None:
    raw_html = source_path.read_text(encoding="utf-8", errors="ignore")
    raw_html = strip_leading_noise(raw_html)
    raw_html = drop_toc_block(raw_html)
    raw_html = strip_trailing_noise(raw_html)

    soup = BeautifulSoup(raw_html, "html.parser")

    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()

    html_fragment = str(soup)
    markdown = md(
        html_fragment,
        heading_style="ATX",
        bullets="-",
        strip=["img"],
    )
    markdown = normalize_markdown(markdown)

    target_path.write_text(markdown, encoding="utf-8")


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    html_files = sorted(INPUT_DIR.glob("*.html"))
    if not html_files:
        print("No .html files found")
        return

    for html_file in html_files:
        output_file = OUTPUT_DIR / f"{html_file.stem}.md"
        convert_one_file(html_file, output_file)
        print(f"Converted: {html_file.name} -> {output_file}")

    print(f"Done. Converted {len(html_files)} files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
