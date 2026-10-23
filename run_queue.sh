#!/bin/bash
# Sequential GPU queue. Hardware safety is enforced INSIDE each Python script
# (adaptivekv/safety.py). If this runner is killed, its children are killed too.
cd /home/abh511/AdaptiveKV-prototype && source .venv/bin/activate
LOG=$1; shift; mkdir -p $LOG
trap 'pkill -TERM -P $$; exit 130' INT TERM HUP USR1 USR2
i=0
for cmd in "$@"; do
  i=$((i+1))
  echo "[$(date +%T)] START $i: $cmd" | tee -a $LOG/queue.log
  bash -c "exec $cmd" > $LOG/step$i.log 2>&1 &
  wait $!; rc=$?
  echo "[$(date +%T)] END $i rc=$rc $(grep -h '^safety:' $LOG/step$i.log)" | tee -a $LOG/queue.log
done
echo "QUEUE FINISHED" | tee -a $LOG/queue.log
