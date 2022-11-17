for (( round=1; round<=10; round+=1 )) 
do
    echo "=====Round $round====="
    python3 auto_train.py
    sleep 5
done