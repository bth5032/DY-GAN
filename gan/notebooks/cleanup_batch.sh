for i in `seq 4 14`; do echo "${i}========="; cat progress/v${i}_batch512_bgbd_mllANDwidth_NonTC_newdata_mllfix/log.txt; done > tmp_runs.txt
while read line; do if [[ $line = "Stats Score"* ]]; then echo -n ${line//Stats Score: /}" "; else echo ${line/#*statistic=/}; fi; done < tmp_runs.txt > tmp_run2.txt
python quickclean.py > tmp_cleaned.txt
sort -n -k1 tmp_cleaned.txt > new_sorted_by_score.txt
sort -n -k2 tmp_cleaned.txt > new_sorted_by_KS.txt

rm tmp_runs.txt tmp_run2.txt tmp_cleaned.txt
