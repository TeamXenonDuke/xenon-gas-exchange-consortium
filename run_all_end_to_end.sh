#!/bin/bash
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

[ -d "testing/expected" ] || mkdir "testing/expected"

# Runs each config file in the testing/config folder with test_end_to_end.py
for config in testing/config/*.py; do
    # Finds subject name from config file
    subject=$(basename "$config" .py)

    # Searched for the matching subject data folder in testing/subjects
    folder=$(find "testing/subjects" -type d -name "${subject}" | head -n 1)

    if [[ "$1" == "batch" ]]; then
        echo -e "${YELLOW}Starting program for subject: '${subject}'...${NC}"
        python main.py --config $config --folder $folder
        mv testing/subjects/${subject}/gx/${subject}_stats.csv testing/expected
    else
        # Searches for the matching expected_csv in the testing/expected folder
        expected_csv=$(find "testing/expected" -type f -name "${subject}*.csv" | head -n 1)

        # Runs pytest with the config, expected csv, and subject data
        if [[ "$subject" == "009-028B" ]]; then
            echo -e "${YELLOW}Starting end-to-end test for subject: '${subject}' (default subject)...${NC}"
            pytest test_end_to_end.py -s
        elif [[ -z "$expected_csv" ]] || [[ -z "$folder" ]]; then
            echo -e "${RED}ERROR: Missing csv and/or subject folder. ${NC}Skipping test for subject: '${subject}'..."
        else
            echo -e "${YELLOW}Starting end-to-end test for subject: '${subject}'...${NC}"
            pytest test_end_to_end.py -s --config=${config} --csv=${expected_csv} --folder=${folder} 
        fi
    fi
done