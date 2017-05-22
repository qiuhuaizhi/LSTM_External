#!/bin/bash
tasknum=1
while [ $tasknum -le 1 ]
do
        echo "Task"${tasknum}"_P"
	python lstm_airline_predict.py "Task"${tasknum}"_P"
        tasknum=$(( $tasknum + 1 ))
done
