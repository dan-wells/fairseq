#!/bin/sh

#set -euo pipefail

AUDIO_ROOT=$1
MANIFEST=$2

if [ $# -ne 2 ]; then
  echo "Usage: $(basename $0) audio_root/ manifest"
  echo "Write audio <manifest> including files under <audio_root>."
  exit 1
fi

printf "${AUDIO_ROOT}\n" > "$MANIFEST"

while read full_path; do
  rel_path=${full_path#"$AUDIO_ROOT"}
  length=$(soxi -s "$full_path" 2>/dev/null)
  if [ -n "$length" ]; then
    printf "${rel_path}\t${length}\n" >> "$MANIFEST"
  fi
done < <(find "$AUDIO_ROOT")
