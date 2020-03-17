sh run_uni.sh $1 $2 $3
wait
mkdir ../result/$1\_lstm/uni_trial0/
mv ../result/$1\_lstm/*.* ../result/$1\_lstm/uni_trial0/

sh run_uni.sh $1 $2 $3
wait
mkdir ../result/$1\_lstm/uni_trial1/
mv ../result/$1\_lstm/*.* ../result/$1\_lstm/uni_trial1/

sh run_uni.sh $1 $2 $3
wait
mkdir ../result/$1\_lstm/uni_trial2/
mv ../result/$1\_lstm/*.* ../result/$1\_lstm/uni_trial2/

sh run_uni.sh $1 $2 $3
wait
mkdir ../result/$1\_lstm/uni_trial3/
mv ../result/$1\_lstm/*.* ../result/$1\_lstm/uni_trial3/

sh run_uni.sh $1 $2 $3
wait
mkdir ../result/$1\_lstm/uni_trial4/
mv ../result/$1\_lstm/*.* ../result/$1\_lstm/uni_trial4/
