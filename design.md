# The Debated Researcher — Design System

## Brand

**Name:** The Debated Researcher

## Colours

| Token          | Value     | Usage                              |
|----------------|-----------|-------------------------------------|
| `--bg`         | `#161d2e` | Page background (deep navy)        |
| `--surface`    | `#1e2740` | Card / panel background            |
| `--surface2`   | `#263050` | Input fields, code blocks          |
| `--border`     | `#2e3a58` | Borders, dividers                  |
| `--accent`     | `#6c63ff` | Primary action colour              |
| `--accent-hover` | `#8179ff` | Hover state for accent           |
| `--text`       | `#e8eaf6` | Body text                          |
| `--text-muted` | `#8a90b0` | Secondary / helper text            |
| `--success`    | `#4caf76` | Completed status                   |
| `--warning`    | `#e8a04a` | Warning / timeout status           |
| `--error`      | `#e85a4a` | Error / failed status              |
| `--running`    | `#5a9be8` | In-progress status                 |

## Typography

| Role          | Family            | Weight / Size     | Loaded from  |
|---------------|-------------------|-------------------|--------------|
| Title (h1)    | **Merriweather**  | 700 / 36px        | Google Fonts |
| Headings / UI | **Work Sans**     | SemiBold 600 / 30px | Google Fonts |
| Body text     | **Merriweather**  | 300–700 / 20px    | Google Fonts |

### Where each font applies

- **Work Sans** (`--font-heading`): labels, section titles, buttons, tabs, agent names, status badges.
- **Merriweather** (`--font-body`): title h1, body copy, inputs, textareas, prose panels, depth hints, token summaries.

## Layout

- Max content width: **680 px**
- Card padding: **32 px**
- Border radius: **10 px**
- Cards have a subtle `box-shadow: 0 1px 3px rgba(0,0,0,0.06)`

## Status badge palette

| Status     | Background              | Text colour      |
|------------|--------------------------|------------------|
| pending    | `rgba(107,107,107,0.15)` | `--text-muted`   |
| running    | `rgba(41,128,185,0.15)`  | `--running`      |
| completed  | `rgba(46,139,87,0.15)`   | `--success`      |
| failed     | `rgba(192,57,43,0.15)`   | `--error`        |
| timed_out  | `rgba(200,123,0,0.15)`   | `--warning`      |
