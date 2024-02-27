```
mdconvert production.dcd -o concatenado.pdb -t step5_input.psf
 python CG_transformation.py -f input_openmm.pdb -m  popc.amber.map -d production.dcd
``
