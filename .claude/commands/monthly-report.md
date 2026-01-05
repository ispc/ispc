# Monthly Report Generator

Generate a monthly report based on all PRs merged to https://github.com/ispc/ispc during the previous month.

allowed-tools: Bash(gh pr *), Bash(gh issue *)

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
   gh issue view N --repo ispc/ispc --json title,body,labels
```

5. **Fetch releases published during the period**:
```bash
   gh release list --repo ispc/ispc --limit 20 \
     --json tagName,publishedAt,url
```

   Filter releases where `publishedAt` falls within the date range. For each release found:
```bash
   gh release view TAG_NAME --repo ispc/ispc --json tagName,publishedAt,url,body
```

   Extract key highlights from release notes (typically 2-3 most important items).

6. **Categorize PRs** using this priority order:

   | Category | Label matches | Content indicators |
   |----------|--------------|-------------------|
   | Features | `enhancement`, `feature`, `new` | New functionality, new targets, new options |
   | Performance | `performance`, `optimization` | Speed, memory, size improvements |
   | Bug fixes | `bug`, `fix`, `bugfix` | Crash fixes, correctness issues |
   | CI/Infrastructure | `CI`, `infrastructure`, `build`, `test` | Workflows, build system, testing |

7. **Generate the report**:
```
ISPC Monthly Report - [Start Date] to [End Date]

Highlights
[2-3 sentences summarizing key themes, major achievements, and notable contributors]
[If a MAJOR release (vX.Y.0) was published during the period, mention it prominently here with release URL and 2-3 key features from release notes]

Features
    • [Author] [what + why + user benefit] ([full PR URL])

Performance
    • [Author] [improvement + measurable impact if available] ([full PR URL])

Bug fixes
    • [Author] [what was broken + user impact + resolution] ([full PR URL])

CI/Infrastructure
    • [Author] [change + why needed] ([full PR URL])

Release Activity
    • [Author] [release-related work] ([full PR URL])
    • [For MINOR/PATCH releases (vX.Y.Z where Z>0): ISPC vX.Y.Z released with brief description ([release URL])]

Work in Progress
    • [Author] [what is being developed + expected benefit] ([full PR URL])
```

## Guidelines

- **Combine related PRs**: Group PRs for the same feature/fix into one bullet with multiple links
- **Focus on user impact**: Explain WHY the change matters to users, not HOW it was implemented. Avoid low-level technical details like specific symbol names, function signatures, or internal APIs.
- **Use issue context**: Enrich descriptions with motivation from linked issues
- **Plain text only**: No markdown formatting—must paste cleanly into Outlook
- **Full URLs always**: https://github.com/ispc/ispc/pull/XXXX
- **Omit empty sections**: Don't include categories with no PRs
- **Author names**: Use real first names, not GitHub usernames
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
    • Arina added avx512gnr* (GraniteRapids) targets, enabling support for Intel's latest server processors (https://github.com/ispc/ispc/pull/3670).

**Combined PRs:**
    • Arina optimized stdlib compilation with a width family system that reduces bitcode duplication, cutting binary size and build time by approximately 30% (https://github.com/ispc/ispc/pull/3617, https://github.com/ispc/ispc/pull/3632, https://github.com/ispc/ispc/pull/3657). The family-based approach now allows adding new targets to ISPC with minimal increase in binary size.

**Work in Progress:**
    • Antoni is finalizing optimization of popcnt implementation for AVX512 resulting in up to 3.5x speedup (https://github.com/ispc/ispc/pull/3639).

**Highlights with major release:**
    ISPC v1.29.0 was released (https://github.com/ispc/ispc/releases/tag/v1.29.0), featuring sample-based profile-guided optimization, optimized dispatcher, new avx512gnr targets for Intel Granite Rapids, and numerous bug fixes and performance improvements. Based on a patched LLVM 20.1.8. Full release notes are available here: https://github.com/ispc/ispc/releases/tag/v1.29.0

**Minor release in Release Activity:**
    • ISPC v1.29.1 released as a hotfix addressing SSE2/SSE4/PS4 regression (https://github.com/ispc/ispc/releases/tag/v1.29.1).
    