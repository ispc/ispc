// Verify Make dependency quoting, including real source and include filenames.
// The Python driver passes tabs and backslashes without shell-dependent quoting.
// RUN: %{python} "%S/Inputs/deps-escaping.py" %{ispc} "%t.dir" host
