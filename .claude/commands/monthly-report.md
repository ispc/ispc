---
description: Generate a monthly report based on all PRs merged to https://github.com/ispc/ispc during the previous month
allowed-tools: Bash(gh pr list *), Bash(gh issue view *), Bash(gh release list *), Bash(gh release view *), Bash(gh api users/*)
---

# Monthly Report Generator

## Instructions

1. **Calculate the rolling 30-day date range**:
   - End date: Yesterday (one day before today)
   - Start date: 31 days before today
   - Example: If today is December 5, the range is November 4 – December 4

2. **Fetch all merged PRs with full details in a single query**:
```bash
   gh pr list --repo ispc/ispc --state merged \
     --search "merged:YYYY-MM-DD..YYYY-MM-DD -author:app/dependabot" \
     --limit 200 \
     --json number,title,body,author,mergedAt,labels,url
```
   
   Notes:
   - The `merged:START..END` syntax ensures proper date bounds
   - The `-author:app/dependabot` filter excludes automated dependency updates
   - If 200 results returned, split date range and run multiple queries

3. **Fetch open PRs (Work in Progress)** created during the period:
```bash
   gh pr list --repo ispc/ispc --state open \
     --search "created:YYYY-MM-DD..YYYY-MM-DD -author:app/dependabot" \
     --limit 50 \
     --json number,title,body,author,createdAt,labels,url
```

4. **Extract and fetch linked issues** when needed:
   
   Scan PR bodies for patterns: `Fixes #N`, `Closes #N`, `Resolves #N`, `Related to #N`
   
   **Only fetch issue details** if:
   - PR body lacks clear motivation/user impact, AND
   - An issue is explicitly linked
```bash
   gh issue view N --repo ispc/ispc --json title,body,labels,author
```

5. **Identify external reporters and contributors** (best effort):

   For each linked issue and each PR authored outside the ISPC core team (skip team members and bots such as `app/github-actions`, `app/dependabot`, `app/copilot-swe-agent`), try to determine which company or project the person represents:
```bash
   gh issue view N --repo ispc/ispc --json author
   gh api users/LOGIN -q '[.name,.company,.email,.blog,.bio]|@tsv'
   gh api users/LOGIN/orgs -q '.[].login'
```

   Sources, in order of reliability: the `company` field, the email domain (ignore generic domains like gmail.com), GitHub organizations, blog/bio, and context in the issue/PR body (e.g., "our engine", product names). Only attribute a company/project when there is clear evidence; if it is unknown, say nothing. Never guess.

6. **Fetch releases published during the period**:
```bash
   gh release list --repo ispc/ispc --limit 20 \
     --json tagName,publishedAt,url
```

   Filter releases where `publishedAt` falls within the date range. For each release found:
```bash
   gh release view TAG_NAME --repo ispc/ispc --json tagName,publishedAt,url,body
```

   Extract key highlights from release notes (typically 2-3 most important items).

7. **Categorize PRs** using this priority order:

   | Category | Label matches | Content indicators |
   |----------|--------------|-------------------|
   | Features | `enhancement`, `feature`, `new` | New functionality, new targets, new options |
   | Performance | `performance`, `optimization` | Speed, memory, size improvements |
   | Bug fixes | `bug`, `fix`, `bugfix` | Crash fixes, correctness issues |
   | CI/Infrastructure | `CI`, `infrastructure`, `build`, `test` | Workflows, build system, testing |

8. **Generate the report**:
```
ISPC Monthly Report - [Start Date] to [End Date]

Highlights
[2-3 sentences summarizing key themes, and major achievements]
[If a MAJOR release (vX.Y.0) was published during the period, mention it prominently here with release URL and 2-3 key features from release notes]

Features
    • [What + why + user benefit] ([full PR URL])

Performance
    • [Improvement + measurable impact if available] ([full PR URL])

Bug fixes
    • [What was broken + user impact + resolution] ([full PR URL])

CI/Infrastructure
    • [Change + why needed] ([full PR URL])

External contributions
    • [What was contributed + user benefit, with the organization if step 5 found one] ([full PR URL])

Release Activity
    • [Release-related work] ([full PR URL])
    • [For MINOR/PATCH releases (vX.Y.Z where Z>0): ISPC vX.Y.Z released with brief description ([release URL])]

Work in Progress
    • [What is being developed + expected benefit] ([full PR URL])
```

## Guidelines

- **Combine related PRs**: Group PRs for the same feature/fix into one bullet with multiple links
- **Focus on user impact**: Explain WHY the change matters to users, not HOW it was implemented. Avoid low-level technical details like specific symbol names, function signatures, or internal APIs.
- **Use issue context**: Enrich descriptions with motivation from linked issues
- **Plain text only**: No markdown formatting—must paste cleanly into Outlook
- **Full URLs always**: https://github.com/ispc/ispc/pull/XXXX
- **Omit empty sections**: Don't include categories with no PRs
- **External contributions section**: PRs authored outside the ISPC core team (see step 5) go in "External contributions" instead of Features/Performance/Bug fixes, so their impact is visible on its own.
- **Don't repeat the previous report**: If the user provides the previous report, drop every PR, issue and release it already covers, even when it falls inside the date range (e.g., a release published on the first days of the window that the previous report already announced). Mention it in the Highlights only if something new happened (e.g., a follow-up patch release).
- **Faceless report**: Do not mention authors, contributors, reviewers, or GitHub usernames anywhere in the report. Describe only what was done.
- **External attribution**: When step 5 identified the company/project behind an externally reported issue or contributed PR, mention the organization, not the person (e.g., "reported by RENDEASE", "contributed by the HardenedBSD project").
- **Output file**: Save to `YYYY_MM_DD_report.txt` using the end date (e.g., `2024_12_04_report.txt`)

## Writing Style

Prefer high-level explanations that focus on user-facing impact over implementation details:

| Bad (implementation-focused) | Good (impact-focused) |
|------------------------------|----------------------|
| Fixed linker errors related to __hash_memory symbol | Resolving linker errors that prevented compilation on Apple platforms |
| Fixed lit tests after recent changes | Ensuring continued compatibility with upcoming LLVM releases |
| Updated VNNI intrinsic arguments to match LLVM 22 API changes | Preparing ISPC for the next LLVM major version |
| Added retry logic and timeout handling | Reducing false-positive build failures |

## Examples

**Single PR:**
    • Added avx512gnr* (GraniteRapids) targets, enabling support for Intel's latest server processors (https://github.com/ispc/ispc/pull/3670).

**Combined PRs:**
    • Optimized stdlib compilation with a width family system that reduces bitcode duplication, cutting binary size and build time by approximately 30% (https://github.com/ispc/ispc/pull/3617, https://github.com/ispc/ispc/pull/3632, https://github.com/ispc/ispc/pull/3657). The family-based approach now allows adding new targets to ISPC with minimal increase in binary size.

**Work in Progress:**
    • Finalizing optimization of popcnt implementation for AVX512 resulting in up to 3.5x speedup (https://github.com/ispc/ispc/pull/3639).

**Highlights with major release:**
    ISPC v1.29.0 was released (https://github.com/ispc/ispc/releases/tag/v1.29.0), featuring sample-based profile-guided optimization, optimized dispatcher, new avx512gnr targets for Intel Granite Rapids, and numerous bug fixes and performance improvements. Based on a patched LLVM 20.1.8. Full release notes are available here: https://github.com/ispc/ispc/releases/tag/v1.29.0

**Minor release in Release Activity:**
    • ISPC v1.29.1 released as a hotfix addressing SSE2/SSE4/PS4 regression (https://github.com/ispc/ispc/releases/tag/v1.29.1).
    