# Project Documentation Guide

This folder contains the project documentation source files for the NBA Career Longevity Prediction project, structured using [MkDocs](http://www.mkdocs.org/).

## Structure

* **`mkdocs.yml`**: Configuration file containing site settings, theme configs, navigation structures, and author details (**Aryan Goel**).
* **`docs/`**: Subfolder holding the source Markdown files:
  * `index.md`: Project summary, modeling pipelines, and general description.
  * `getting-started.md`: Environment setup, pipeline run instructions, and testing details.

---

## How to Build and Serve Locally

To build and view the documentation website locally, follow these steps:

1. **Install MkDocs** (already included in `requirements.txt`):
   ```bash
   pip install mkdocs
   ```

2. **Serve the website locally**:
   ```bash
   # Start the local development server
   mkdocs serve
   ```
   *Open `http://127.0.0.1:8000/` in your browser to view the live documentation site.*

3. **Build the static HTML site**:
   ```bash
   # Compile Markdown into static HTML files
   mkdocs build
   ```
   *This compiles the site into the `site/` folder, ready for deployment.*
