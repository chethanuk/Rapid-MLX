function flush_pending(    i) {
  for (i = 1; i <= pending_count; i++) print pending[i]
  pending_count = 0
}
function tail_after_close(line,    p) {
  p = index(line, "-->")
  return substr(line, p + 3)
}
function rest_is_clean(s,    o, c) {
  while (1) {
    if (s ~ /^[[:space:]]*$/) return 1
    if (s !~ /^[[:space:]]*<!--/) return 0
    o = index(s, "<!--"); s = substr(s, o + 4)
    c = index(s, "-->"); if (c == 0) return 0
    s = substr(s, c + 3)
  }
}
function indented_code(line) {
  return (line ~ /^    / || line ~ /^\t/)
}
function container_payload(line,    s, before) {
  s = line
  sub(/^   /, "", s); sub(/^  /, "", s); sub(/^ /, "", s)
  while (1) {
    before = s
    while (s ~ /^>[[:space:]]*/) sub(/^>[[:space:]]*/, "", s)
    if (s ~ /^[-+*][[:space:]]+/) sub(/^[-+*][[:space:]]+/, "", s)
    else if (s ~ /^[0-9]+[.)][[:space:]]+/) sub(/^[0-9]+[.)][[:space:]]+/, "", s)
    if (s == before) break
  }
  return s
}
function fence_marker(line,    s, ch, n) {
  if (indented_code(line)) return ""
  s = line
  sub(/^ */, "", s)
  if (s ~ /^```/) ch = "`"
  else if (s ~ /^~~~/) ch = "~"
  else return ""
  n = 0
  while (substr(s, n + 1, 1) == ch) n++
  return substr(s, 1, n)
}
in_comment {
  pending[++pending_count] = $0
  if ($0 ~ /-->/) {
    in_comment = 0
    if (rest_is_clean(tail_after_close($0))) pending_count = 0
    else flush_pending()
  }
  next
}
!in_fence {
  fm = fence_marker($0)
  if (fm != "") {
    print
    fence_char = substr(fm, 1, 1); fence_len = length(fm); in_fence = 1
    next
  }
}
in_fence {
  print
  fm = fence_marker($0)
  if (fm != "" && substr(fm, 1, 1) == fence_char && length(fm) >= fence_len \
      && $0 ~ ("^[[:space:]]*[" fence_char "]+[[:space:]]*$")) in_fence = 0
  next
}
!indented_code($0) && container_payload($0) ~ /^<!--/ {
  if ($0 ~ /-->/) {
    if (!rest_is_clean(tail_after_close($0))) print
    next
  }
  in_comment = 1
  pending[++pending_count] = $0
  next
}
{ print }
END {
  # An unmatched opener can hide every generated line that follows in GitHub's
  # Markdown renderer. Drop its buffered tail; preflight then rejects a file or
  # section that contains no other visible content.
  if (!in_comment) flush_pending()
}
