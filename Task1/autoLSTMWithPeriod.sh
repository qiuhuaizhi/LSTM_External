#!/bin/bash
tasknum=1
while [ $tasknum -le 15 ]
do
        echo "Task"${tasknum}"_P"
	python lstm_external_predict.py "Task"${tasknum}"_P"
        tasknum=$(( $tasknum + 1 ))
done
