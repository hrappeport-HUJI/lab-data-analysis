#!/bin/bash
# Move the heavy raw part of a finished experiment to the deep archive (Google Drive only), keeping the
# experiment folder in ~/Research/Raw_data with its notebooks, results and a DEEP_ARCHIVE.md pointer.
#
#   bash ~/Research/Code/lab-data-analysis/raw_data/deep_archive.sh "<experiment folder name>" <subfolder>
#   e.g. bash ~/Research/Code/lab-data-analysis/raw_data/deep_archive.sh "2026-11-04 PI uptake of KLY and TBL8 by microscopy" microscopy
#
# Steps: 1) SHA-256 manifest of the subfolder (kept in the experiment folder), 2) copy to
# My Drive/Research deep archive/Raw_data/<experiment>/<subfolder>, 3) check every file arrived with the right size,
# 4) write DEEP_ARCHIVE.md. It never deletes: it prints the delete command, to run once Google Drive shows the
# upload as finished. Afterwards update the experiment's Zettelkasten note (source_refs: storage_root Google Drive).
set -euo pipefail
EXP="$1"; SUB="${2%/}"
RD="$HOME/Research/Raw_data"
SRC="$RD/$EXP/$SUB"
ARCH="$HOME/Library/CloudStorage/GoogleDrive-hrappeport@gmail.com/My Drive/Research deep archive/Raw_data/$EXP/$SUB"
MAN="$RD/$EXP/DEEP_ARCHIVE_${SUB//\//_}.sha256"
[ -d "$SRC" ] || { echo "No such folder: $SRC"; exit 1; }
[ -d "$HOME/Library/CloudStorage/GoogleDrive-hrappeport@gmail.com/My Drive" ] || { echo "Google Drive (hrappeport@gmail.com) is not mounted"; exit 1; }
n=$(find "$SRC" -type f ! -name .DS_Store | wc -l | tr -d ' '); gb=$(du -sk "$SRC" | awk '{printf "%.1f", $1/1048576}')
df_gb=$(df -k "$HOME" | awk 'NR==2 {printf "%d", $4/1048576}')
echo "Archiving $n files, $gb GB. Free disk: $df_gb GB (Drive caches files locally until they are uploaded)."
need=$(du -sk "$SRC" | awk '{printf "%d", $1/1048576 + 10}')
if [ "$df_gb" -lt "$need" ]; then
  echo "Not enough free disk: Drive needs a local copy until the upload finishes (about $need GB incl. 10 GB margin)."
  echo "Archive one subfolder at a time instead, e.g.:  bash $0 \"$EXP\" \"$SUB/<subfolder>\"  and remove each local copy once uploaded."
  exit 1
fi
echo "1/4 checksums -> $MAN"
(cd "$SRC" && find . -type f ! -name .DS_Store -print0 | sort -z | xargs -0 nice -n 10 shasum -a 256) > "$MAN"
echo "2/4 copying to Drive"
mkdir -p "$ARCH"
nice -n 10 rsync -a --exclude .DS_Store "$SRC/" "$ARCH/"
echo "3/4 checking sizes"
bad=$( (cd "$SRC" && find . -type f ! -name .DS_Store -exec stat -f '%z %N' {} +) | while read -r size path; do
  s2=$(stat -f %z "$ARCH/$path" 2>/dev/null || echo missing); [ "$s2" = "$size" ] || echo "$path"; done | wc -l | tr -d ' ')
[ "$bad" = 0 ] || { echo "ERROR: $bad files missing or different on Drive; nothing else done."; exit 1; }
echo "4/4 writing $RD/$EXP/DEEP_ARCHIVE.md"
cat >> "$RD/$EXP/DEEP_ARCHIVE.md" <<EOF
## $SUB/ — deep-archived $(date +%F)

- Moved to: Google Drive, account hrappeport@gmail.com, \`My Drive/Research deep archive/Raw_data/$EXP/$SUB/\`
- Contents: $n files, $gb GB. Checksums: \`$(basename "$MAN")\` in this folder (\`shasum -a 256 -c\` from inside the folder restores the check).
- To work on it again: in Finder, right-click the folder in Google Drive → "Available offline", or copy it back here.

EOF
echo
echo "Copied and verified. When the Google Drive menu-bar icon shows no pending uploads, remove the local copy with:"
echo "  rm -rf \"$SRC\""
echo "Then update the Zettelkasten note's source_refs (storage_root: Google Drive, relative_path as above)."
