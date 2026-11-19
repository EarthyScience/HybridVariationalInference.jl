#!/usr/bin/env bash
# Use sacctmgr or scontrol to inspect the QOS. In Slurm, QOS CPU limits usually appear in fields such as:
#     MaxTRES=cpu=N — max CPUs per job under that QOS
#     MaxTRESPU=cpu=N — max CPUs per user
#     GrpTRES=cpu=N — max total CPUs for all jobs using that QOS
#     GrpTRESMins=cpu=N — total CPU-minutes
#     GrpTRESRunMins=cpu=N — running CPU-minutes

set -euo pipefail

account="${1:-mbg_cpu}"

# Get all unique QOS names associated with the account.
qos_list=$(
  sacctmgr -n -P show assoc account="$account" format=QOS |
    tr ',' '\n' |
    sed 's/^[[:space:]]*//; s/[[:space:]]*$//' |
    grep -v -E '^(|\(null\)|None)$' |
    sort -u
)

if [[ -z "$qos_list" ]]; then
  echo "No QOS found for account $account" >&2
  exit 1
fi

format="Name,MaxTRES,MaxTRESPU,GrpTRES,GrpTRESMins,GrpTRESRunMins"
names="MaxTRES MaxTRESPU GrpTRES GrpTRESMins GrpTRESRunMins"

while IFS= read -r qos; do
  [[ -z "$qos" ]] && continue

  #echo "QOS: $qos"  # only print if CPU related limit found

  line=$(sacctmgr -n -P show qos name="$qos" format="$format" || true)

  if [[ -z "$line" ]]; then
    echo "  (QOS not found or no data)"
    continue
  fi

  out=$(
    printf '%s\n' "$line" | awk -F'|' -v names_str="$names" '
      BEGIN { split(names_str, names, " ") }
      {
        for (i = 2; i <= NF; i++) {
          n = split($i, parts, ",")
          found = 0
          for (j = 1; j <= n; j++) {
            if (parts[j] ~ /^cpu=/) {
              if (!found) {
                printf "  %s: ", names[i-1]
                found = 1
              } else {
                printf ", "
              }
              printf "%s", parts[j]
            }
          }
          if (found) print ""
        }
      }'
  )

  if [[ -n "$out" ]]; then
    echo "QOS: $qos"
    printf '%s\n' "$out"
#   else
#     echo "  (no CPU-related limits found)"
  fi

  echo
done <<< "$qos_list"