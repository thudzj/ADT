run ADT_{EXP} with: python adt_exp.py --batch-size 64 --step-size 0.3 --loss-type gaussian --entropy 0.01 
run ADT_{EXP-AM} with: python adt_expam.py --batch-size 64 --dist gaussian --entropy 0.01
run ADT_{IMP-AM} with: python adt_impam.py --model-dir implicit_ent1_bs64_th0.9 --batch-size 64 --entropy 1. --entropy_th 0.9
