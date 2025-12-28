# Binding affinity prediction of ~51,000 small molecules for ASGPR by EDDY

<p align="center">
  <img src="https://github.com/Lee1701/Lee2024a/blob/main/etc/Lee-NAIRR240089-Bridges2.png" width="600">
</p>

This repository contains results from the NAIRR Pilot project,<br>
https://nairrpilot.org/projects/awarded?_requestNumber=NAIRR240089

The major goal of the project was to carry out virtual screening of a large-scale compound library to identify high-affinity ASGPR ligands to improve MoDEs, molecular degraders of extracellular proteins, by our pretrained AI models (EDDY),<br>
https://pubs.acs.org/doi/10.1021/acs.jcim.4c01116

There are a total of 1,112 sequence-based EDDY models in 5 model families: 6 models in D1 family, 6 in D2, 300 in D1F, 300 in D2F, and 500 in D3.

Using resources on PSC Bridges-2 Regular Memory (~100,000 CPU SUs), we successfully completed predicting ~31.8 million binding affinities for ~51,000 small molecules from the Therma Fisher Maybridge Screening Library targeting ASGPR by our 622 sequence-based EDDY models. Due to the limited resources, we used 10 D3 alternative models instead of 500 D3 original models. Full predictions by the 500 D3 models were later completed on the McCleary cluster at the Yale Center for Research Computing.

[Contact]<br>
- William H. Lee, PhD: ho-joon.lee%at%yale.edu
