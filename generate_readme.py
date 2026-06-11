import os
import datetime as dt


def generate_readme(notes_dir="notes"):
    if not os.path.isdir(notes_dir):
        print(f"Error: Directory '{notes_dir}' not found.")
        return ""

    md_files = [
        f for f in sorted(os.listdir(notes_dir))
        if f.endswith(".md")
    ]

    if not md_files:
        print("No markdown files found.")
        return ""

    lines = []
    for f in md_files:
        title = f.replace("-", " ").replace("_", " ")[:-3].title()
        link = f"{notes_dir}/{f}"
        lines.append(f"- [{title}]({link})")

    date_str = dt.datetime.now().strftime("%Y-%m-%d")

    readme = (
        f"# ML Course Notes\n\n"
        f"Last updated: {date_str}\n\n"
        "---\n\n"
        "## 📒 Notes\n\n"
        + "\n".join(lines)
        + "\n"
    )

    return readme


if __name__ == "__main__":
    readme_content = generate_readme("notes")

    if readme_content:
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
        print("README.md updated successfully.")