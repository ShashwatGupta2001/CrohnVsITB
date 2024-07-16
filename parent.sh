#!/bin/bash

./runner_z_ts_fr.sh output.txt 2>&1
echo "REACHED END SUCCESSFULLY"
touch "REACHED_END.txt"