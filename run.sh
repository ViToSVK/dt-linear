#!/bin/bash
#python experiments.py games_randomltl_naive debug replace
#python experiments.py games_wash_permissive debug replace
python experiments.py games_randomltl_encoded debug replace
python experiments.py games_wash debug replace
python experiments.py mdps_coBuchiAS_randomltl debug replace
python experiments.py mdps_prism_consensus debug replace
python experiments.py mdps_prism_csma debug replace
python experiments.py mdps_prism_leader debug replace
python experiments.py mdps_prism_mer debug replace
