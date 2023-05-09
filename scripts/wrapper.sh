#!/bin/zsh

# Send to slack channel that the command is started
emoji=:kobugi:
slack.sh $emoji \`start!\` $emoji \`\`\`$@\`\`\`

# Run the specified command
$@ 2> >(tee /tmp/log >&2)

# Get the exit status of $@ 
exit_bash="${PIPESTATUS[0]}" exit_zsh="${pipestatus[1]}"
exit=$exit_bash$exit_zsh

output=`cat /tmp/log`

# Check the exit code of the command
if [ $exit -ne 0 ]; then
    emoji=:anger:
    slack.sh $emoji$emoji \`ERROR\` $emoji$emoji \`\`\`$@ \`\`\` \`\`\`$output\`\`\` 
    exit 1
fi

emoji=:doge:
slack.sh $emoji \`done\` $emoji \`\`\`$@\`\`\`
exit 0




