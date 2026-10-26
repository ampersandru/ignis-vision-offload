#!/usr/bin/env bash
# The behavioural test of .githooks/commit-msg (`make commit-msg-selftest`),
# which is docs/agents/testing.md's rule applied to a script. The hook reads
# only the message file git hands it, so each case is a message in a temp file
# and the hook's exit status: no repository, nothing committed.

set -uo pipefail
hook="$(dirname "$0")/../.githooks/commit-msg"
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
failed=0

# expect accept|refuse <name> <message>
expect() {
    printf '%s' "$3" > "$dir/msg"
    if sh "$hook" "$dir/msg" 2>/dev/null; then got=accept; else got=refuse; fi
    if [ "$got" = "$1" ]; then
        printf 'ok    %s\n' "$2"
    else
        printf 'FAIL  %s: expected %s, got %s\n' "$2" "$1" "$got"
        failed=1
    fi
}

expect accept "a one-paragraph message ending with its issue" \
    "The server downloads a catalog entry in the background (#376)"
expect accept "a subject and a body, the body ending with the issue" \
    "$(printf 'Short subject\n\nThe body says why, and ends with the issue (#376)\n')"
expect accept "trailing blank lines and git's comment lines" \
    "$(printf 'The change and its reason (#376)\n\n# Please enter the commit message\n')

"
expect accept "a merge that says what it brings in" \
    "Merge #374: the server downloads and removes models from the Server view (#374)"
expect accept "a catch-up merge with a message of its own" \
    "Merge 374-integration into the T2 branch before the report (#377)"
expect accept "the release commit mk/version.sh prints" \
    "ignis 0.9.0"

expect refuse "no issue at the end" \
    "The server downloads a catalog entry in the background"
expect refuse "an issue named mid-message only" \
    "Follows #376 with the cancel route"
expect refuse "git's default merge message" \
    "Merge branch '374-integration' into 374-t2"
expect refuse "git's default merge message with an issue appended" \
    "Merge branch '374-integration' into 374-t2 (#377)"
expect refuse "git's default message for a remote-tracking merge" \
    "Merge remote-tracking branch 'origin/main' into speech-diarizer"
expect refuse "a Co-Authored-By trailer" \
    "$(printf 'The change and its reason (#376)\n\nCo-Authored-By: Someone <someone@example.com>\n')"
expect refuse "a Co-Authored-By trailer in lower case" \
    "$(printf 'The change and its reason (#376)\n\nco-authored-by: Someone <someone@example.com>\n')"
expect refuse "a release-shaped subject with more after it" \
    "$(printf 'ignis 0.9.0\n\nand a body with no issue\n')"

exit $failed
