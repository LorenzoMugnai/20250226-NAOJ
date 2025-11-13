#!/bin/bash

if [ $# -lt 3 ]; then
    echo "Usage: $0 <number_of_threads> <init_start> <init_end>"
    exit 1
fi

# set - e 

# Numero di threads specificato dall'utente
num_threads=$1
init_start=$2
init_end=$3

# Percorsi dei file e impostazioni iniziali
output_for="/home/ariel/lorenzo.mugnai/USER_DATA/Alfnoor2/20250226 - NAOJ/first_week/forwards"
output_ret="/home/ariel/lorenzo.mugnai/USER_DATA/Alfnoor2/20250226 - NAOJ/first_week/retrievals"
target_list="/home/ariel/lorenzo.mugnai/USER_DATA/Alfnoor2/20250226 - NAOJ/first_week/Mtype_MetaData.csv"
input_for="/home/ariel/lorenzo.mugnai/USER_DATA/Alfnoor2/20250226 - NAOJ/first_week/forward_LM.par"
input_ret="/home/ariel/lorenzo.mugnai/USER_DATA/Alfnoor2/20250226 - NAOJ/first_week/inverse.par"

# Calcola il numero totale di target nel file della lista dei target
total_targets=$((init_end - init_start + 1))
echo "Total targets: $total_targets"

# Determina il numero di target per thread
targets_per_thread=$((total_targets / num_threads))
extra_targets=$((total_targets % num_threads)) # Residuo della divisione
echo "Targets per thread: $targets_per_thread"

start=0
end=0

echo "Starting forward processing..."
for (( i=0; i<num_threads; i++ )); do
    start=$((init_start + i * targets_per_thread + (i < extra_targets ? i : extra_targets) ))
    
    if [ $i -lt $extra_targets ]; then
        end=$((start + targets_per_thread))
    else
        end=$((start + targets_per_thread - 1))
    fi

    echo "Processing targets $start to $end in forward mode"
    nice -n 10 alfnoor-forward -s -t "$target_list" -n "$start-$end" -i "$input_for" -o "$output_for" &
done

wait  # Aspetta che tutti i processi di alfnoor-forward finiscano prima di iniziare il retrieval

echo "All forward processing complete. Starting inverse retrieval..."

for (( i=0; i<num_threads; i++ )); do
    start=$((init_start + i * targets_per_thread + (i < extra_targets ? i : extra_targets) ))
    
    if [ $i -lt $extra_targets ]; then
        end=$((start + targets_per_thread))
    else
        end=$((start + targets_per_thread - 1))
    fi

    echo "Processing targets $start to $end in inverse retrieval"
    nice -n 10 alfnoor-inverse -f -t "$target_list" -n "$start-$end" -i "$input_ret" -o "$output_ret" &
done

wait  # Aspetta che tutti i processi di alfnoor-inverse finiscano

echo "All processing complete."
