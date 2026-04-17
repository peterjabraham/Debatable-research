# The Debated Researcher — Design System

## Brand

**Name:** The Debated Researcher

## Colours

| Token          | Value     | Usage                              |
|----------------|-----------|-------------------------------------|
| `--bg`         | `#f2f0ec` | Page background (warm off-white)   |
| `--surface`    | `#ffffff` | Card / panel background            |
| `--surface2`   | `#e8e5df` | Input fields, code blocks          |
| `--border`     | `#d4d0c8` | Borders, dividers                  |
| `--accent`     | `#6c63ff` | Primary action colour              |
| `--accent-hover` | `#5a52e0` | Hover state for accent           |
| `--text`       | `#2c2c2c` | Body text                          |
| `--text-muted` | `#6b6b6b` | Secondary / helper text            |
| `--success`    | `#2e8b57` | Completed status                   |
| `--warning`    | `#c87b00` | Warning / timeout status           |
| `--error`      | `#c0392b` | Error / failed status              |
| `--running`    | `#2980b9` | In-progress status                 |

## Typography

| Role      | Family                | Weight      | Loaded from          |
|-----------|-----------------------|-------------|----------------------|
| Headings  | **Ubuntu**            | 400 / 500 / 700 | Google Fonts    |
| Body text | **Kreon**             | 300–700     | Google Fonts         |

### Where each font applies

- **Ubuntu** (`--font-heading`): `h1`, labels, section titles, buttons, tabs, agent names, status badges.
- **Kreon** (`--font-body`): body copy, inputs, textareas, prose panels, depth hints, token summaries.

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
