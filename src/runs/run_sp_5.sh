sh run_sp.sh $1 $2 $3
mkdir ../result/$1/sp_trial0/
mv ../result/$1/*.* sp_trial0/
sh run_sp.sh $1 $2 $3
mkdir ../result/$1/sp_trial1/
mv ../result/$1/*.* sp_trial1/
sh run_sp.sh $1 $2 $3
mkdir ../result/$1/sp_trial2/
mv ../result/$1/*.* sp_trial2/
sh run_sp.sh $1 $2 $3
mkdir ../result/$1/sp_trial3/
mv ../result/$1/*.* sp_trial3/
sh run_sp.sh $1 $2 $3
mkdir ../result/$1/sp_trial4/
mv ../result/$1/*.* sp_trial4/
