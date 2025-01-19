start_time=$(date +"%s")
sleep 2
end_time=$(date +"%s")
elapsed_time=$((end_time - start_time))
echo $elapsed_time
