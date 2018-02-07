folder_path=`pwd` 
input_dir=${folder_path}/data/test/content-test.txt  
reference_dir=${folder_path}/data/test/title-test.txt  
summary_dir=${folder_path}/data/test/summary.txt  
   
python predict.py $input_dir $reference_dir $summary_dir  