#!/bin/bash

# Combined script that extracts functions from LLVM IR files and generates HTML diff reports
# Combines functionality from extract_func.sh and gen_diff_html.sh

# Check if all parameters are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 function_name in_dir out_dir"
    echo "  function_name: Name of the function to extract"
    echo "  in_dir: Directory containing .ll files (see instructions below)"
    echo "  out_dir: Directory where extracted functions will be saved and diff report generated"
    echo ""
    echo "To generate input_dir with ISPC optimization passes:"
    echo "  1. Create a test ISPC file (e.g., test.ispc)"
    echo "  2. Run ISPC with debug output to generate .ll files:"
    echo "     ispc --target=avx2-i32x8 --opt=O2 --debug-phase=pre:last --dump-file=dbg test.ispc -o test.o"
    echo "  3. Use the 'dbg' directory as input_dir for this script"
    echo ""
    echo "Example:"
    echo "  $0 my_function ./dbg ./extracted_func"
    exit 1
fi

function_name="$1"
in_dir="$2"
out_dir="$3"

# Validate and normalize input directory path
if [ ! -d "$in_dir" ]; then
    echo "Error: Input directory '$in_dir' does not exist"
    exit 1
fi
in_dir=$(realpath "$in_dir")

# Validate and normalize output directory path
if ! out_dir=$(realpath "$out_dir" 2>/dev/null); then
    # If realpath fails (directory doesn't exist), try to create parent and get realpath
    parent_dir=$(dirname "$out_dir")
    if [ ! -d "$parent_dir" ]; then
        echo "Error: Parent directory '$parent_dir' does not exist"
        exit 1
    fi
    mkdir -p "$out_dir" || {
        echo "Error: Failed to create output directory '$out_dir'"
        exit 1
    }
    out_dir=$(realpath "$out_dir")
fi

# Ensure output directory exists and is writable
if [ ! -d "$out_dir" ]; then
    mkdir -p "$out_dir" || {
        echo "Error: Failed to create output directory '$out_dir'"
        exit 1
    }
fi

if [ ! -w "$out_dir" ]; then
    echo "Error: Output directory '$out_dir' is not writable"
    exit 1
fi

echo "=== Step 1: Extracting function '$function_name' ==="

# Change to input directory
cd "$in_dir" || exit 1

# Find all .ll files and process them
find . -name "*.ll" -type f | while read -r file; do
    # Create necessary subdirectories in output directory
    out_subdir="$out_dir/$(dirname "$file")"
    mkdir -p "$out_subdir"

    # Extract the specified function and save to output directory
    if llvm-extract --func="$function_name" "$file" -S -o "$out_dir/$file" 2>/dev/null; then
        echo "Successfully processed: $file"
    else
        echo "Error processing: $file (function may not exist)"
    fi
done

cd - >/dev/null || exit 1
echo "Function extraction complete. Results saved in $out_dir"

echo ""
echo "=== Step 2: Generating HTML diff report ==="

# Change to output directory for diff generation
cd "$out_dir" || exit 1

COMBINED_DIFF="combined.diff"
OUTPUT_HTML="diffreport.html"
# Skip first 5 lines to ignore LLVM IR header/metadata when comparing
# LLVM IR files typically start with target triple, data layout, and module attributes
# that are not relevant for function-level optimization comparisons
# Note: tail -n +6 actually skips the first 5 lines (starts from line 6)
SKIP_LINES=6

rm -f "$COMBINED_DIFF" "$OUTPUT_HTML"
touch "$COMBINED_DIFF"

# Check if we have any .ll files
if ! ls ir*.ll >/dev/null 2>&1; then
    echo "Error: No ir*.ll files found in directory '$out_dir'"
    exit 1
fi

# Get sorted list of .ll files once to avoid repeated execution using array for safe handling
# Use bash arrays for safe filename handling with special characters
sorted_files=()
while IFS= read -r -d '' file; do
    sorted_files+=("$(basename "$file")")
done < <(find . -maxdepth 1 -name 'ir*.ll' -print0 | sort -zV)

# Check if we found any files to process
if [ ${#sorted_files[@]} -eq 0 ]; then
    echo "Error: No ir*.ll files found for processing"
    exit 1
fi

for i in "${!sorted_files[@]}"; do
    f="${sorted_files[$i]}"
    # Get next file in the array
    if (( i + 1 < ${#sorted_files[@]} )); then
        n="${sorted_files[$((i + 1))]}"
    else
        continue
    fi

    # Extract numeric indices from filenames with validation
    c_num=$(echo "$f" | sed -E 's/[^0-9]*([0-9]+).*/\1/')
    n_num=$(echo "$n" | sed -E 's/[^0-9]*([0-9]+).*/\1/')

    # Validate that we successfully extracted numeric values
    if ! [[ "$c_num" =~ ^[0-9]+$ ]] || ! [[ "$n_num" =~ ^[0-9]+$ ]]; then
        echo "Warning: Could not extract numeric indices from files '$f' and '$n', skipping comparison"
        continue
    fi

    if [[ $((c_num + 1)) -eq $n_num ]]; then
        # Only diff if the contents actually differ (skipping LLVM IR header/metadata)
        if ! cmp -s <(tail -n +$SKIP_LINES "$f") <(tail -n +$SKIP_LINES "$n"); then
            echo "### Diff $f vs $n" >> "$COMBINED_DIFF"

            # Use --label (or -L) to set the filename in the diff output
            diff -u \
                 --label="$f" \
                 --label="$n" \
                 <(tail -n +$SKIP_LINES "$f") \
                 <(tail -n +$SKIP_LINES "$n") >> "$COMBINED_DIFF"

            echo "" >> "$COMBINED_DIFF"
        fi
    fi
done

# Check if we have any diffs to report
if [ ! -s "$COMBINED_DIFF" ]; then
    echo "No differences found between consecutive optimization passes."
    exit 0
fi

# Generate the HTML with diff2html (rtfpessoa)
# Using side-by-side style, feel free to omit "-s side" for line-by-line
if ! command -v diff2html >/dev/null 2>&1; then
    echo "Error: diff2html not found."
    echo "For installation instructions, see: https://github.com/rtfpessoa/diff2html-cli"
    exit 1
fi

if diff2html -i file -o stdout -s side -- "$COMBINED_DIFF" > "$OUTPUT_HTML"; then
    echo "HTML diff report generated: $out_dir/$OUTPUT_HTML"
else
    echo "Error: diff2html command failed. HTML report may be incomplete or corrupted."
    exit 1
fi

cd - >/dev/null || exit 1

echo ""
echo "=== Complete ==="
echo "Function '$function_name' extracted and diff report generated in '$out_dir'"
